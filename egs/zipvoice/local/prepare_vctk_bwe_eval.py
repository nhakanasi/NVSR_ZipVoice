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
Build a held-out VCTK evaluation set for the mel bandwidth extender.

This is the out-of-domain counterpart to prepare_bwe_eval.py. The extender and
the ZipVoice weights were both fine-tuned on LibriTTS, so a LibriTTS dev-clean
result cannot distinguish "the extender restores bandwidth" from "the model
learned this corpus". VCTK is a different corpus in every way that matters here:
different speakers, a different microphone and room, many non-American accents,
and read newspaper sentences rather than audiobook prose.

VCTK 0.92 ships at 48 kHz. Prompts are resampled to 24 kHz first, so the clean
reference matches the rate ZipVoice works at, and the degraded copies are then
derived from that 24 kHz signal exactly as in the LibriTTS recipe.

Only the prompt audio is ever needed: the target utterance contributes its
transcript and nothing else. That is what makes streaming practical -- roughly
80 members out of a 11.7 GB archive.

The corpus is read through zipfile's API, so --vctk-zip accepts either a local
VCTK-Corpus-0.92.zip or an https URL. The URL form needs `remotezip`, which
fetches individual members over HTTP range requests instead of downloading the
whole archive.

Usage:

    python egs/zipvoice/local/prepare_vctk_bwe_eval.py \
        --vctk-zip https://datashare.ed.ac.uk/bitstream/handle/10283/3443/\
VCTK-Corpus-0.92.zip \
        --out-dir /c/zipvoice_data/data/vctk_bwe_eval \
        --num-utts 76
"""

import argparse
import io
import logging
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
import torchaudio

# Same prompt-length window as the LibriTTS eval set, so the two are comparable.
MIN_PROMPT_SECONDS = 3.0
MAX_PROMPT_SECONDS = 10.0

# A target sentence has to be long enough for word error rate to mean something.
MIN_TARGET_CHARS = 40

# ZipVoice works at 24 kHz; the clean reference has to be at that rate or the
# similarity score would measure a resampling difference.
NATIVE_RATE = 24000

# VCTK utterances 001-024 are the same elicitation paragraphs for every speaker.
# Reusing text across speakers would correlate the WER of unrelated conditions,
# and those recordings are also the least representative of the corpus.
FIRST_SPEAKER_SPECIFIC_UTT = 25

# p315 is missing its transcripts in the 0.92 release, and p280's mic1 channel
# is documented as unreliable.
EXCLUDED_SPEAKERS = {"p280", "p315"}

AUDIO_MEMBER = "wav48_silence_trimmed/{spk}/{utt}_mic1.flac"
TEXT_MEMBER = "txt/{spk}/{utt}.txt"


@contextmanager
def open_corpus(location: str) -> Iterator["object"]:
    """Open a VCTK zip from a path or an https URL under one API."""
    if re.match(r"^https?://", location):
        try:
            from remotezip import RemoteZip
        except ImportError as e:
            raise SystemExit(
                "Reading VCTK over HTTP needs `pip install remotezip`. "
                "Alternatively pass a local VCTK-Corpus-0.92.zip."
            ) from e
        with RemoteZip(location) as z:
            yield z
    else:
        import zipfile

        with zipfile.ZipFile(location) as z:
            yield z


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--vctk-zip",
        type=str,
        required=True,
        help="""Path to VCTK-Corpus-0.92.zip, or its https URL. A URL is read
        with HTTP range requests and only the members actually used are
        transferred, which is a few tens of megabytes rather than 11.7 GB.
        """,
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
        default=76,
        help="""How many prompt/target pairs to build. The default matches the
        LibriTTS eval set so the two are directly comparable.
        """,
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=2,
        help="Cap on pairs drawn from one speaker.",
    )
    parser.add_argument(
        "--candidates-per-speaker",
        type=int,
        default=14,
        help="""How many utterances of a speaker to consider. Each one costs a
        small transcript fetch; only those that survive text filtering cost an
        audio fetch.
        """,
    )
    parser.add_argument(
        "--rates",
        type=str,
        default="8000,16000,22050,24000",
        help="Comma-separated prompt sampling rates. 24000 is the control.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for speaker and utterance choice.",
    )
    return parser


def list_speakers(zf) -> Dict[str, List[str]]:
    """Map speaker -> sorted utterance ids that have both audio and text."""
    audio, text = defaultdict(set), defaultdict(set)
    for name in zf.namelist():
        if name.endswith("_mic1.flac"):
            spk = name.split("/")[-2]
            audio[spk].add(name.split("/")[-1][: -len("_mic1.flac")])
        elif name.startswith("txt/") and name.endswith(".txt"):
            spk = name.split("/")[-2]
            text[spk].add(name.split("/")[-1][: -len(".txt")])
    return {
        spk: sorted(ids & text[spk])
        for spk, ids in audio.items()
        if spk not in EXCLUDED_SPEAKERS and (ids & text[spk])
    }


def speaker_specific(utt_ids: List[str]) -> List[str]:
    """Drop the elicitation paragraphs shared by every speaker."""
    kept = []
    for utt in utt_ids:
        try:
            index = int(utt.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if index >= FIRST_SPEAKER_SPECIFIC_UTT:
            kept.append(utt)
    return kept


def read_text(zf, spk: str, utt: str) -> str:
    with zf.open(TEXT_MEMBER.format(spk=spk, utt=utt)) as f:
        return f.read().decode("utf-8").strip()


def read_audio_24k(zf, spk: str, utt: str) -> Tuple[torch.Tensor, float]:
    """Load one utterance and resample it to 24 kHz mono."""
    with zf.open(AUDIO_MEMBER.format(spk=spk, utt=utt)) as f:
        raw = io.BytesIO(f.read())
    audio, rate = torchaudio.load(raw)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if rate != NATIVE_RATE:
        audio = torchaudio.functional.resample(
            audio, orig_freq=rate, new_freq=NATIVE_RATE
        )
    return audio, audio.size(1) / NATIVE_RATE


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

    rng = random.Random(args.seed)
    rows: List[Dict[str, str]] = []

    with open_corpus(args.vctk_zip) as zf:
        by_speaker = list_speakers(zf)
        logging.info(f"VCTK has {len(by_speaker)} usable speakers")
        speakers = sorted(by_speaker)
        rng.shuffle(speakers)

        # Every speaker contributes its first pair before anyone contributes a
        # second, so a set truncated by --num-utts is still as broad as it can be.
        for round_idx in range(args.max_per_speaker):
            for spk in speakers:
                if len(rows) >= args.num_utts:
                    break
                used = {r["prompt_utt"] for r in rows if r["speaker"] == spk} | {
                    r["target_utt"] for r in rows if r["speaker"] == spk
                }
                candidates = [
                    u for u in speaker_specific(by_speaker[spk]) if u not in used
                ]
                if len(candidates) < 2:
                    continue
                rng.shuffle(candidates)
                candidates = candidates[: args.candidates_per_speaker]

                texts = {}
                for utt in candidates:
                    try:
                        texts[utt] = read_text(zf, spk, utt)
                    except KeyError:
                        continue

                # The target only ever contributes its transcript, so filter it
                # on text alone and spend no bandwidth on its audio. VCTK
                # sentences are much shorter than LibriTTS ones, so take the
                # longest rather than the first: more reference words per
                # utterance is free WER resolution.
                eligible = (
                    u for u in candidates if len(texts.get(u, "")) >= MIN_TARGET_CHARS
                )
                targets = sorted(eligible, key=lambda u: len(texts[u]), reverse=True)
                if not targets:
                    continue

                # The prompt needs audio, and its duration is only knowable
                # after fetching it, so try candidates until one fits.
                prompt_utt, prompt_audio = None, None
                for utt in candidates:
                    if utt not in texts:
                        continue
                    try:
                        audio, duration = read_audio_24k(zf, spk, utt)
                    except (KeyError, RuntimeError):
                        continue
                    if MIN_PROMPT_SECONDS <= duration <= MAX_PROMPT_SECONDS:
                        prompt_utt, prompt_audio = utt, audio
                        break
                if prompt_utt is None:
                    logging.info(f"{spk}: no candidate in the prompt-length window")
                    continue

                target_utt = next((u for u in targets if u != prompt_utt), None)
                if target_utt is None:
                    continue

                wav_name = f"{spk}_{target_utt}"
                clean_path = clean_dir / f"{wav_name}.wav"
                torchaudio.save(str(clean_path), prompt_audio, NATIVE_RATE)

                degraded: Dict[int, Path] = {}
                for rate in rates:
                    path = out_dir / "prompts" / f"sr{rate}" / f"{wav_name}.wav"
                    if rate == NATIVE_RATE:
                        # An untouched copy is the control. Running it through a
                        # resampler anyway would put a transition band in the one
                        # condition that is supposed to have none.
                        resampled = prompt_audio
                    else:
                        resampled = torchaudio.functional.resample(
                            prompt_audio, orig_freq=NATIVE_RATE, new_freq=rate
                        )
                    torchaudio.save(str(path), resampled, rate)
                    degraded[rate] = path

                rows.append(
                    {
                        "speaker": spk,
                        "prompt_utt": prompt_utt,
                        "target_utt": target_utt,
                        "wav_name": wav_name,
                        "prompt_text": texts[prompt_utt],
                        "clean_prompt": str(clean_path),
                        "text": texts[target_utt],
                        **{f"prompt_{r}": str(p) for r, p in degraded.items()},
                    }
                )
                logging.info(f"[{len(rows)}/{args.num_utts}] {wav_name}")

    if len(rows) < args.num_utts:
        logging.warning(
            f"Only {len(rows)} of the requested {args.num_utts} pairs could be built."
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


if __name__ == "__main__":
    main()
