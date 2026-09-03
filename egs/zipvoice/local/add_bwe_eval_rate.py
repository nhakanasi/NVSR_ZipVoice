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
Add a prompt sampling rate to an evaluation set that already exists.

`prepare_bwe_eval.py` and `prepare_vctk_bwe_eval.py` draw their utterance pairs
at random. Re-running either with an extra entry in --rates would therefore
rebuild the whole set, and every number already measured against the old set
would no longer be comparable. This derives the new rate from the clean prompts
already on disk instead, so the utterances, the pairing and the target texts are
untouched and the new condition slots into the existing tables.

The resampling matches `prepare_bwe_eval.py` exactly: one
`torchaudio.functional.resample` from the clean prompt's native rate, written
out at the target rate.

Usage::

    python3 local/add_bwe_eval_rate.py --eval-dir data/bwe_eval --rate 4000
"""
import argparse
import logging
import os
from pathlib import Path

import torchaudio


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add one prompt rate to an eval set.")
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="An eval set built by prepare_bwe_eval.py or its VCTK counterpart.",
    )
    parser.add_argument(
        "--rate", type=int, required=True, help="Prompt sampling rate to add."
    )
    parser.add_argument(
        "--template-rate",
        type=int,
        default=None,
        help="""Existing test list to copy the utterance rows from. Defaults to
        whichever test_sr*.tsv is found first; any of them carries the same rows.
        """,
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    eval_dir = Path(args.eval_dir)
    clean_dir = eval_dir / "prompts" / "clean"
    if not clean_dir.is_dir():
        raise SystemExit(f"No clean prompts in {clean_dir}")

    out_dir = eval_dir / "prompts" / f"sr{args.rate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav in sorted(clean_dir.glob("*.wav")):
        audio, native_rate = torchaudio.load(str(wav))
        if args.rate == native_rate:
            # Matching prepare_bwe_eval.py: the control is an untouched copy,
            # because running it through a resampler would introduce the one
            # transition band the control is meant not to have.
            resampled = audio
        else:
            resampled = torchaudio.functional.resample(
                audio, orig_freq=native_rate, new_freq=args.rate
            )
        torchaudio.save(str(out_dir / wav.name), resampled, args.rate)

    # The test list differs from any other rate's only in the prompt path, so
    # copying one keeps the rows, their order and the texts identical.
    if args.template_rate is not None:
        template = eval_dir / f"test_sr{args.template_rate}.tsv"
    else:
        found = sorted(eval_dir.glob("test_sr*.tsv"))
        if not found:
            raise SystemExit(f"No test_sr*.tsv to copy rows from in {eval_dir}")
        template = found[0]
    old_tag = template.stem.replace("test_", "")

    dest = eval_dir / f"test_sr{args.rate}.tsv"
    count = 0
    with open(template, encoding="utf-8") as src, open(
        dest, "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in src:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            # Paths in these lists are absolute and Windows-style; swap only the
            # rate directory so the rest of the path survives untouched.
            head, tail = os.path.split(parts[2])
            parts[2] = os.path.join(os.path.dirname(head), f"sr{args.rate}", tail)
            out.write("\t".join(parts) + "\n")
            count += 1
    logging.info(
        f"Wrote {count} rows to {dest}, rows copied from {template.name} ({old_tag})"
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
