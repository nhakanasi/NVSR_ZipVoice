#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu)
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
Extract the natural recording of each evaluation target sentence.

Mel-cepstral distortion needs a reference rendition of the same words, and the
evaluation sets do not carry one: they were built to score speaker similarity,
word error rate and naturalness, none of which needs the target audio. The
target utterance id survives in the file name, though, because both preparation
scripts name their rows `{speaker}_{target utterance id}`, so the reference can
be recovered from the corpus after the fact without rebuilding the eval set.

Everything is written at 24 kHz to match the synthesised audio, so the MCD
script never has to resample.

Usage, LibriTTS:

    python egs/zipvoice/local/prepare_mcd_references.py \
        --corpus libritts \
        --eval-dir data/bwe_eval \
        --cuts data/fbank/libritts_cuts_dev-clean.jsonl.gz

Usage, VCTK (a URL is read with HTTP range requests, as in the eval prep):

    python egs/zipvoice/local/prepare_mcd_references.py \
        --corpus vctk \
        --eval-dir data/vctk_bwe_eval \
        --vctk-zip https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
"""  # noqa: E501

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import List

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_vctk_bwe_eval import (  # noqa: E402
    AUDIO_MEMBER,
    open_corpus,
    read_audio_24k,
)

TARGET_RATE = 24000


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        choices=["libritts", "vctk"],
        help="Which corpus the evaluation set was built from.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="""Evaluation directory holding test_score.tsv. References are
        written to its `references` subdirectory.
        """,
    )
    parser.add_argument(
        "--cuts",
        type=str,
        default=None,
        help="LibriTTS only: the lhotse cuts the eval set was drawn from.",
    )
    parser.add_argument(
        "--vctk-zip",
        type=str,
        default=None,
        help="VCTK only: path or https URL of VCTK-Corpus-0.92.zip.",
    )
    return parser


def read_wav_names(eval_dir: Path) -> List[str]:
    names = []
    with open(eval_dir / "test_score.tsv", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                names.append(line.split("\t")[0])
    return names


def target_id(wav_name: str) -> str:
    """Strip the speaker prefix both preparation scripts prepend."""
    return wav_name.split("_", 1)[1]


def extract_libritts(names: List[str], cuts_path: Path, out_dir: Path) -> int:
    wanted = {target_id(n): n for n in names}
    written = 0
    opener = gzip.open if cuts_path.suffix == ".gz" else open
    with opener(cuts_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cut = json.loads(line)
            name = wanted.pop(cut["id"], None)
            if name is None:
                continue
            recording = cut["recording"]
            rate = recording["sampling_rate"]
            audio, loaded = torchaudio.load(
                recording["sources"][0]["source"],
                frame_offset=int(round(cut["start"] * rate)),
                num_frames=int(round(cut["duration"] * rate)),
            )
            assert loaded == rate, (loaded, rate)
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if rate != TARGET_RATE:
                audio = torchaudio.functional.resample(audio, rate, TARGET_RATE)
            torchaudio.save(str(out_dir / f"{name}.wav"), audio, TARGET_RATE)
            written += 1
    if wanted:
        logging.warning(f"{len(wanted)} target cuts were not found in the manifest")
    return written


def extract_vctk(names: List[str], location: str, out_dir: Path) -> int:
    written = 0
    with open_corpus(location) as zf:
        for name in names:
            utt = target_id(name)
            speaker = utt.split("_")[0]
            member = AUDIO_MEMBER.format(spk=speaker, utt=utt)
            try:
                zf.getinfo(member)
            except KeyError:
                logging.warning(f"{member} is not in the archive")
                continue
            audio, _ = read_audio_24k(zf, speaker, utt)
            torchaudio.save(str(out_dir / f"{name}.wav"), audio, TARGET_RATE)
            written += 1
    return written


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    torch.set_num_threads(1)

    eval_dir = Path(args.eval_dir)
    out_dir = eval_dir / "references"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = read_wav_names(eval_dir)
    logging.info(f"{len(names)} evaluation rows")

    if args.corpus == "libritts":
        assert args.cuts is not None, "--cuts is required for LibriTTS"
        written = extract_libritts(names, Path(args.cuts), out_dir)
    else:
        assert args.vctk_zip is not None, "--vctk-zip is required for VCTK"
        written = extract_vctk(names, args.vctk_zip, out_dir)

    logging.info(f"Wrote {written} references to {out_dir}")


if __name__ == "__main__":
    main()
