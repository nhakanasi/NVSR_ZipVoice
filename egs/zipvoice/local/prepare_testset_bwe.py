#!/usr/bin/env python3
"""
Band-limit an upstream ZipVoice test set so the bandwidth study can be run on it.

`prepare_bwe_eval.py` builds an evaluation set from a lhotse cuts manifest; this
script does the same job for the three test sets shipped with the paper --
Seed-TTS test-en, Seed-TTS test-zh and LibriSpeech-PC test-clean -- which arrive
as a ready-made `test.tsv` of

    wav_name <TAB> prompt_text <TAB> prompt_wav <TAB> text

and therefore need no speaker pairing, only degradation. The two scripts agree on
what comes out, so downstream tooling does not care which built the set:

    test_sr{rate}.tsv   the degraded prompt. Synthesis reads this.
    test_score.tsv      the original prompt. Scoring reads this, because scoring
                        against a degraded reference measures the degradation
                        rather than the synthesis.

Prompt paths in the upstream lists are relative to `egs/zipvoice`; everything
written here is absolute, so the lists work from any working directory.

Subsampling exists because the cascade baseline is the budget: AudioSR restores
at roughly six times real time, and the full three sets carry 2124 distinct
prompts. A seeded subsample of 200 rows holds the standard error of a SIM-o mean
near 0.006 -- an order below the effects being measured -- for a twentieth of the
compute. Pass `--num-utts 0` to take the set whole.

Usage:

    python egs/zipvoice/local/prepare_testset_bwe.py \
        --test-list download/seedtts_testset/en/test.tsv \
        --out-dir /c/zipvoice_data/data/seedtts_en_bwe \
        --num-utts 200 --rates 8000,16000
"""

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import List

import torchaudio


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        required=True,
        help="Upstream test.tsv: wav_name, prompt_text, prompt_wav, text.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write degraded prompts and test lists into.",
    )
    parser.add_argument(
        "--transcript-list",
        type=Path,
        default=None,
        help="""LibriSpeech-PC ships a second list of normalised transcripts for
        word error rate. When given, it is subsampled to the same rows and
        written as transcript.tsv beside the others.""",
    )
    parser.add_argument(
        "--num-utts",
        type=int,
        default=200,
        help="Rows to keep. 0 keeps the whole set.",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default="8000,16000",
        help="Comma-separated prompt sampling rates to build.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory the relative prompt paths in --test-list resolve against.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subsampling seed.")
    return parser


def read_tsv(path: Path) -> List[List[str]]:
    with open(path, encoding="utf-8") as handle:
        # QUOTE_NONE: the transcripts contain bare double quotes, which the
        # default dialect reads as field quoting -- it silently merges rows and
        # strips the marks.
        return [
            row
            for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            if row
        ]


def write_tsv(path: Path, rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
        ).writerows(rows)
    logging.info(f"Wrote {len(rows)} lines to {path}")


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    rates = [int(r) for r in args.rates.split(",") if r.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(args.test_list)
    logging.info(f"Read {len(rows)} rows from {args.test_list}")

    if args.num_utts and args.num_utts < len(rows):
        # Sorted indices, not shuffled rows: the subsample keeps the upstream
        # order, so two runs with the same seed are diffable line by line.
        keep = sorted(random.Random(args.seed).sample(range(len(rows)), args.num_utts))
        rows = [rows[i] for i in keep]
        logging.info(f"Subsampled to {len(rows)} rows with seed {args.seed}")

    if args.transcript_list is not None:
        wanted = {row[0] for row in rows}
        transcripts = [r for r in read_tsv(args.transcript_list) if r[0] in wanted]
        if len(transcripts) != len(rows):
            logging.warning(
                f"{len(transcripts)} transcripts for {len(rows)} rows; word error "
                "rate will be computed over the intersection."
            )
        write_tsv(args.out_dir / "transcript.tsv", transcripts)

    for rate in rates:
        (args.out_dir / "prompts" / f"sr{rate}").mkdir(parents=True, exist_ok=True)

    # Prompts repeat across rows -- Seed-TTS test-en has 666 distinct prompts over
    # 1088 rows -- so each one is degraded once and reused. This matters mostly
    # for the cascade, which pays a diffusion pass per distinct prompt file.
    degraded: dict = {}
    out_rows: dict = {rate: [] for rate in rates}
    score_rows: List[List[str]] = []

    for index, row in enumerate(rows):
        wav_name, prompt_text, prompt_wav, text = row[0], row[1], row[2], row[3]
        source = Path(prompt_wav)
        if not source.is_absolute():
            source = (args.root / prompt_wav).resolve()
        score_rows.append([wav_name, prompt_text, str(source), text])

        if source not in degraded:
            audio, native_rate = torchaudio.load(str(source))
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            paths = {}
            for rate in rates:
                path = args.out_dir / "prompts" / f"sr{rate}" / f"{source.stem}.wav"
                if rate == native_rate:
                    # No resampling: running the control through a resampler
                    # anyway would put a transition band in the one condition
                    # that is supposed to have none.
                    resampled = audio
                else:
                    resampled = torchaudio.functional.resample(
                        audio, orig_freq=native_rate, new_freq=rate
                    )
                torchaudio.save(str(path), resampled, rate)
                paths[rate] = path
            degraded[source] = paths
            if (index + 1) % 50 == 0:
                logging.info(f"Degraded {len(degraded)} prompts ({index + 1} rows)")

        for rate in rates:
            out_rows[rate].append(
                [wav_name, prompt_text, str(degraded[source][rate]), text]
            )

    for rate in rates:
        write_tsv(args.out_dir / f"test_sr{rate}.tsv", out_rows[rate])
    write_tsv(args.out_dir / "test_score.tsv", score_rows)
    logging.info(
        f"{len(degraded)} distinct prompts degraded to {rates}. Synthesise with "
        "test_sr{rate}.tsv, score with test_score.tsv."
    )


if __name__ == "__main__":
    main()
