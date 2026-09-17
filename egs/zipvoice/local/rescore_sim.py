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
Backfill per-utterance SIM-o files for conditions that were scored before
`zipvoice.eval.speaker_similarity.sim` learned to write them.

`local/bwe_stats.py sim` needs a `sim_scores.tsv` beside each condition's audio
to run its paired tests, and the arms already synthesised predate that file.
Rescoring them one directory at a time through the evaluator's own command line
would reload WavLM-large once per condition -- roughly half the wall clock for
fifty-odd conditions, all of it spent on a model whose weights never change.
This driver holds the evaluator open instead and walks the directories.

It only rescores; no audio is generated and nothing already on disk is
overwritten unless `--overwrite` is passed, so an interrupted pass resumes by
being run again.

Every condition is scored against the same `test_score.tsv`, which is what makes
the arms comparable in the first place and is why one loaded model suffices.

Example::

    python3 local/rescore_sim.py \
        --results-dir /c/zipvoice_data/exp/results \
        --test-list /c/zipvoice_data/data/bwe_eval/test_score.tsv \
        --model-dir /c/zipvoice_data/tts_eval_models
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

from zipvoice.eval.speaker_similarity.sim import SpeakerSimilarity


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write per-utterance SIM-o files for already-synthesised arms."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory holding one subdirectory of audio per condition.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        required=True,
        help="The test list every condition is scored against.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local path of the evaluation model repository, as passed to "
        "zipvoice.eval.speaker_similarity.sim.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        default=None,
        help="Condition subdirectory names. Default: every subdirectory of "
        "--results-dir that holds at least one file of the given extension.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )
    parser.add_argument(
        "--score-name",
        type=str,
        default="sim_scores.tsv",
        help="File written inside each condition directory. Must match what "
        "local/bwe_stats.py sim reads. Default: sim_scores.tsv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescore conditions that already have a score file.",
    )
    return parser


def discover(results_dir: str, extension: str):
    out = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        if any(f.endswith(f".{extension}") for f in os.listdir(path)):
            out.append(name)
    return out


def main() -> None:
    args = get_parser().parse_args()

    conditions = args.conditions or discover(args.results_dir, args.extension)
    if not conditions:
        raise SystemExit(f"No condition directories found under {args.results_dir}")

    todo = []
    for cond in conditions:
        out_path = os.path.join(args.results_dir, cond, args.score_name)
        if os.path.exists(out_path) and not args.overwrite:
            logging.info(f"Skipping {cond}; {args.score_name} exists")
            continue
        todo.append((cond, out_path))

    if not todo:
        logging.info("Nothing to do; every condition already has a score file.")
        return

    logging.info(f"Scoring {len(todo)} of {len(conditions)} conditions")

    sv_model_path = os.path.join(
        args.model_dir, "speaker_similarity/wavlm_large_finetune.pth"
    )
    ssl_model_path = os.path.join(args.model_dir, "speaker_similarity/wavlm_large/")
    if not os.path.exists(sv_model_path) or not os.path.exists(ssl_model_path):
        logging.error(
            "Please download evaluation models from "
            "https://huggingface.co/k2-fsa/TTS_eval_models"
            " and pass this dir with --model-dir"
        )
        sys.exit(1)

    evaluator = SpeakerSimilarity(
        sv_model_path=sv_model_path, ssl_model_path=ssl_model_path
    )

    # A condition whose audio is incomplete raises out of ``score`` rather than
    # scoring a subset, and one bad arm should not cost the pass the model load
    # for all the others, so failures are collected and reported at the end.
    failed = []
    for cond, out_path in todo:
        wav_path = os.path.join(args.results_dir, cond)
        try:
            scores = evaluator.score(wav_path, args.extension, args.test_list)
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            logging.error(f"{cond}: {e}")
            failed.append(cond)
            continue
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("wav_name\tsim\n")
            for name, value in scores.items():
                f.write(f"{name}\t{value:.6f}\n")
        mean = float(np.mean(list(scores.values())))
        logging.info(f"{cond}: SIM-o {mean:.3f} over {len(scores)} utterances")

    if failed:
        logging.error(f"{len(failed)} condition(s) failed: {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
