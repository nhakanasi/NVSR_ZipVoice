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
Build the "plug and play" checkpoint: the trained bandwidth extender in front of
the *released* ZipVoice weights.

The joint recipe fine-tunes the TTS weights alongside the extender, so a
bwe-vs-stock comparison cannot say whether the extender would help a ZipVoice
that was never fine-tuned. This script answers that by taking the released state
dict and adding only the `bwe.*` tensors from a trained ZipVoiceBWE checkpoint.

The two key sets are disjoint and together exhaust the ZipVoiceBWE parameters,
which the script asserts, so the result is a valid ZipVoiceBWE and needs no
inference changes -- run it with `--model-name zipvoice_bwe` like any other.

Note that at a 24 kHz prompt this checkpoint is the same *function* as stock
ZipVoice: the TTS tensors are bit-identical and `band_limit` is a no-op at a
Nyquist cutoff, so the Eq. 10 composite copies every observed bin through. The
two arms still produce different audio because the extender code path consumes
the sampling RNG differently, which makes the pair a useful null condition.

Example::

    python3 local/make_plugplay_checkpoint.py \
        --stock-dir download/zipvoice \
        --bwe-dir exp/zipvoice_bwe \
        --checkpoint-name iter-15000-avg-2.pt \
        --output-dir exp/zipvoice_bwe_plugplay
"""
import argparse
import filecmp
import logging
import os
import shutil

import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a trained bandwidth extender onto stock ZipVoice."
    )
    parser.add_argument(
        "--stock-dir",
        type=str,
        required=True,
        help="Directory holding the released model.pt, model.json and tokens.txt.",
    )
    parser.add_argument(
        "--bwe-dir",
        type=str,
        required=True,
        help="Experiment directory of the trained ZipVoiceBWE checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="iter-15000-avg-2.pt",
        help="Checkpoint to take the bwe.* parameters from. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the merged checkpoint, model.json and tokens.txt to.",
    )
    return parser


def load_state_dict(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"] if "model" in obj else obj


def main() -> None:
    args = get_parser().parse_args()

    stock = load_state_dict(os.path.join(args.stock_dir, "model.pt"))
    trained = load_state_dict(os.path.join(args.bwe_dir, args.checkpoint_name))

    bwe_keys = {k for k in trained if k.startswith("bwe.")}
    tts_keys = set(trained) - bwe_keys
    if tts_keys != set(stock):
        raise SystemExit(
            "The non-bwe parameters of the trained checkpoint do not match the "
            "released model. The two were not trained from the same architecture, "
            f"so the merge would be meaningless ({len(tts_keys)} vs {len(stock)} "
            "tensors)."
        )

    # The extender must come from the trained run and the TTS from the released
    # model; a silent overlap would defeat the whole point of the arm.
    merged = {k: v.clone() for k, v in stock.items()}
    merged.update({k: trained[k].clone() for k in bwe_keys})

    # The tokenizer has to be the one the released checkpoint was trained with,
    # otherwise the text embedding table does not match the weights.
    tokens = os.path.join(args.bwe_dir, "tokens.txt")
    stock_tokens = os.path.join(args.stock_dir, "tokens.txt")
    if not filecmp.cmp(tokens, stock_tokens, shallow=False):
        raise SystemExit(
            "Token files differ between the trained run and the released model."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    # model.json has to be the BWE one: it carries the "bwe" and "bwe_disc"
    # blocks the model class needs to construct the extender.
    shutil.copy(os.path.join(args.bwe_dir, "model.json"), args.output_dir)
    shutil.copy(tokens, args.output_dir)
    out = os.path.join(args.output_dir, args.checkpoint_name)
    torch.save({"model": merged}, out)

    changed = sum(1 for k in stock if not torch.equal(stock[k], trained[k]))
    logging.info(
        f"Wrote {out}: {len(bwe_keys)} extender tensors from {args.checkpoint_name}, "
        f"{len(stock)} TTS tensors from the released model "
        f"({changed} of which joint fine-tuning had modified)."
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
