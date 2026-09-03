#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Build the evaluation set for the mel bandwidth extender.

The extender exists so that a prompt recorded below 24 kHz still conditions
ZipVoice well. Testing that needs prompts that are genuinely band-limited, so
this script takes held-out 24 kHz utterances, resamples each one down to a
target rate and writes it out at that rate. The inference path resamples it back
up, which is exactly what a user supplying a low-rate file gets: the high band
is gone and no amount of interpolation brings it back.

Two kinds of test list come out of this, and the difference matters:

  test_sr{rate}.tsv  points at the degraded prompt. This is what synthesis reads.
  test_score.tsv     points at the original 24 kHz prompt. This is what
                     speaker-similarity scoring reads, because scoring a
                     generated wav against a degraded reference measures the
                     degradation rather than the synthesis.

Both carry the same `wav_name` and the same target text, so a single generated
directory is scored against the clean reference regardless of which rate
produced it.

Usage:

    python egs/zipvoice/local/prepare_bwe_eval.py \
        --cuts data/fbank/libritts_cuts_dev-clean.jsonl.gz \
        --out-dir data/bwe_eval \
        --num-utts 50
"""

import argparse
import gzip
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

# Prompts shorter than this carry too little speaker evidence; longer ones are
# mostly wasted compute and drift away from the length ZipVoice sees at training.
MIN_PROMPT_SECONDS = 3.0
MAX_PROMPT_SECONDS = 10.0

# A target sentence has to be long enough for word error rate to mean something.
MIN_TARGET_CHARS = 40


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cuts",
        type=str,
        required=True,
        help="Lhotse cuts of the held-out set, e.g. the dev-clean manifest.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write prompts and test lists into.",
    )
    parser.add_argument(
        "--num-utts",
        type=int,
        default=50,
        help="How many prompt/target pairs to build.",
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=1,
        help="""Cap on pairs drawn from one speaker. Spreading over speakers
        keeps a single voice from dominating the average, but a held-out set
        with few speakers (dev-clean has 40) cannot reach a useful sample size
        at one apiece, so raise this rather than shrink the evaluation.
        """,
    )
    parser.add_argument(
        "--rates",
        type=str,
        default="8000,16000,22050,24000",
        help="""Comma-separated prompt sampling rates to build. 24000 is the
        undegraded control and must not regress: it is the guard against the
        extender damaging the case it was never needed for.
        """,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for speaker and utterance choice.",
    )
    return parser


def load_cuts(path: Path) -> List[dict]:
    """Read a possibly gzipped lhotse cuts manifest into memory."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_speaker(cuts: List[dict]) -> Dict[str, List[dict]]:
    by_speaker: Dict[str, List[dict]] = defaultdict(list)
    for cut in cuts:
        supervisions = cut.get("supervisions") or []
        if len(supervisions) != 1:
            # Multi-supervision cuts have no single prompt transcript.
            continue
        by_speaker[supervisions[0]["speaker"]].append(cut)
    return by_speaker


def read_audio(cut: dict) -> torch.Tensor:
    """Load one cut's audio at its native rate as (1, num_samples)."""
    recording = cut["recording"]
    source = recording["sources"][0]["source"]
    sampling_rate = recording["sampling_rate"]
    offset = int(round(cut["start"] * sampling_rate))
    num_frames = int(round(cut["duration"] * sampling_rate))
    audio, loaded_rate = torchaudio.load(
        source, frame_offset=offset, num_frames=num_frames
    )
    assert loaded_rate == sampling_rate, (loaded_rate, sampling_rate)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    rates = [int(r) for r in args.rates.split(",") if r.strip()]
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "prompts" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        (out_dir / "prompts" / f"sr{rate}").mkdir(parents=True, exist_ok=True)

    cuts = load_cuts(Path(args.cuts))
    logging.info(f"Read {len(cuts)} cuts from {args.cuts}")

    by_speaker = group_by_speaker(cuts)
    rng = random.Random(args.seed)

    # One pair per speaker: the similarity score is only meaningful when the
    # generated speech and the reference prompt come from the same person, and
    # spreading over speakers stops a single voice dominating the average.
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)

    # Draw every speaker's first pair before anyone's second, so that a set
    # truncated by --num-utts is still as broad as it can be.
    pairs: List[tuple] = []
    for round_idx in range(args.max_per_speaker):
        for speaker in speakers:
            utts = sorted(by_speaker[speaker], key=lambda c: c["id"])
            prompts = [
                c
                for c in utts
                if MIN_PROMPT_SECONDS <= c["duration"] <= MAX_PROMPT_SECONDS
            ]
            targets = [
                c for c in utts if len(c["supervisions"][0]["text"]) >= MIN_TARGET_CHARS
            ]
            if not prompts or len(targets) < 2:
                continue
            prompt_cut = rng.choice(prompts)
            # The target has to be a different utterance, otherwise the model is
            # asked to reproduce the very audio it was conditioned on, and it
            # has to be one this speaker has not already contributed.
            used = {t["id"] for s, p, t in pairs if s == speaker}
            candidates = [
                c
                for c in targets
                if c["id"] != prompt_cut["id"] and c["id"] not in used
            ]
            if not candidates:
                continue
            pairs.append((speaker, prompt_cut, rng.choice(candidates)))

    rows: List[Dict[str, str]] = []
    for speaker, prompt_cut, target_cut in pairs:
        if len(rows) >= args.num_utts:
            break
        wav_name = f"{speaker}_{target_cut['id']}"
        audio = read_audio(prompt_cut)
        native_rate = prompt_cut["recording"]["sampling_rate"]

        clean_path = clean_dir / f"{wav_name}.wav"
        torchaudio.save(str(clean_path), audio, native_rate)

        degraded: Dict[int, Path] = {}
        for rate in rates:
            path = out_dir / "prompts" / f"sr{rate}" / f"{wav_name}.wav"
            if rate == native_rate:
                # No resampling: an untouched copy is the control, and running
                # it through a resampler anyway would put a transition band in
                # the one condition that is supposed to have none.
                resampled = audio
            else:
                resampled = torchaudio.functional.resample(
                    audio, orig_freq=native_rate, new_freq=rate
                )
            torchaudio.save(str(path), resampled, rate)
            degraded[rate] = path

        rows.append(
            {
                "wav_name": wav_name,
                "prompt_text": prompt_cut["supervisions"][0]["text"],
                "clean_prompt": str(clean_path),
                "text": target_cut["supervisions"][0]["text"],
                **{f"prompt_{rate}": str(p) for rate, p in degraded.items()},
            }
        )

    if len(rows) < args.num_utts:
        logging.warning(
            f"Only {len(rows)} of the requested {args.num_utts} pairs could be "
            f"built; {len(speakers)} speakers were available."
        )

    def write_list(path: Path, prompt_key: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(
                    "\t".join(
                        [
                            row["wav_name"],
                            row["prompt_text"],
                            row[prompt_key],
                            row["text"],
                        ]
                    )
                    + "\n"
                )
        logging.info(f"Wrote {len(rows)} lines to {path}")

    for rate in rates:
        write_list(out_dir / f"test_sr{rate}.tsv", f"prompt_{rate}")
    write_list(out_dir / "test_score.tsv", "clean_prompt")

    logging.info(
        "Synthesise with test_sr{rate}.tsv, score with test_score.tsv. "
        "Scoring against the clean prompt is deliberate: a degraded reference "
        "would measure the degradation instead of the synthesis."
    )


if __name__ == "__main__":
    main()
