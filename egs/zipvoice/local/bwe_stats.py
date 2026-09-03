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
Paired statistics over the per-utterance score files the evaluators leave behind.

The headline numbers of this study are means over 76 utterances, and a mean is
not enough to decide anything: it cannot say whether a gap between two arms is
larger than the noise between two runs of the same system, and it cannot
distinguish an arm that lifts its typical output from one that removes a handful
of bad clips. Both questions are answerable without resynthesis and without a
GPU, because `zipvoice.eval.mos.dnsmos` and `zipvoice.eval.wer.seedtts` both
write per-utterance records to disk. This script reads those records back.

Two subcommands, one per record format:

`tail` -- lower-tail analysis of DNSMOS. For a deployment the worst clip matters
more than the average one, because that is the clip a listener notices. Three
views per condition: the 10th percentile, lower quartile and minimum, which say
how bad the bad cases are; the count of utterances below a fixed bar, which is
the number a listener would reject, where the bar is the baseline's own 10th
percentile at the same rate and corpus so that it adapts to the corpus rather
than fixing an arbitrary MOS value; and a paired t-test restricted to the lower
quartile of the baseline, which asks whether an arm specifically repairs the
cases the baseline handles worst. That selection uses baseline scores only, so
it cannot bias the sign of the difference.

`wer` -- paired comparison of Seed-TTS word error rates. Two aggregations are
printed. Seed-TTS WER is the unweighted mean of the per-utterance rates, which
is what the official Seed-TTS evaluation uses. Corpus WER is total errors over
total reference words, which weights long utterances more and is the more stable
of the two. Per-utterance WER is heavily zero-inflated, so a Wilcoxon
signed-rank test is printed beside the paired t.

Examples::

    python3 local/bwe_stats.py tail --dnsmos-dir /c/zipvoice_data/dnsmos

    python3 local/bwe_stats.py wer \
        --results-dir exp/results \
        --conditions stock_sr8000 bwe_sr8000 \
        --baseline stock_sr8000
"""
import argparse
import logging
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired statistics over saved per-utterance scores."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tail = sub.add_parser(
        "tail", help="Lower-tail analysis of per-utterance DNSMOS scores."
    )
    tail.add_argument(
        "--dnsmos-dir",
        type=str,
        required=True,
        help="Directory of <corpus>_<arm>_sr<rate>.tsv score files.",
    )
    tail.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=["libritts", "vctk"],
        help="Corpus prefixes of the score files. Default: %(default)s",
    )
    tail.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=["stock", "plugplay", "bwe_bypass_full", "bwe"],
        help="Arms to report, baseline first. Default: %(default)s",
    )
    tail.add_argument(
        "--rates",
        type=int,
        nargs="+",
        default=[8000, 16000, 22050, 24000],
        help="Prompt sampling rates. Default: %(default)s",
    )
    tail.add_argument(
        "--metric",
        type=str,
        default="P808_MOS",
        help="Column of the score file to analyse. Default: %(default)s",
    )

    wer = sub.add_parser(
        "wer", help="Paired WER comparison over saved seedtts decodes."
    )
    wer.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory holding one subdirectory per condition.",
    )
    wer.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        required=True,
        help="Condition subdirectory names, each containing a decode.txt.",
    )
    wer.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="""Condition to compare every other one against. Defaults to the
        first entry of --conditions.
        """,
    )
    return parser


def load_dnsmos(dnsmos_dir, corpus, arm, rate, metric):
    """Per-utterance scores keyed by wav name, or None if the file is absent."""
    path = os.path.join(dnsmos_dir, "%s_%s_sr%d.tsv" % (corpus, arm, rate))
    if not os.path.exists(path):
        return None
    out = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        col = f.readline().rstrip("\n").split("\t").index(metric)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > col:
                out[parts[0]] = float(parts[col])
    return out


def aligned(a, b):
    """The two score vectors over the utterances both conditions produced."""
    keys = [k for k in a if k in b]
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def run_tail(args) -> None:
    baseline = args.arms[0]

    for corpus in args.corpora:
        print("")
        print("==== %s: lower tail of %s ====" % (corpus, args.metric))
        print(
            "%-6s %-18s %7s %7s %7s %7s %6s"
            % ("rate", "arm", "mean", "p10", "p25", "min", "n<bar")
        )
        for rate in args.rates:
            base = load_dnsmos(
                args.dnsmos_dir, corpus, baseline, rate, args.metric
            )
            if base is None:
                continue
            bar = float(np.percentile(np.array(list(base.values())), 10))
            for arm in args.arms:
                cur = load_dnsmos(
                    args.dnsmos_dir, corpus, arm, rate, args.metric
                )
                if cur is None:
                    continue
                v = np.array(list(cur.values()))
                print(
                    "%-6d %-18s %7.3f %7.3f %7.3f %7.3f %6d"
                    % (
                        rate,
                        arm,
                        v.mean(),
                        np.percentile(v, 10),
                        np.percentile(v, 25),
                        v.min(),
                        int((v < bar).sum()),
                    )
                )
            print("%-6d bar (%s p10) = %.3f" % (rate, baseline, bar))

        print("")
        print(
            "==== %s: does an arm repair the baseline's worst cases? ===="
            % corpus
        )
        print(
            "%-6s %-30s %8s %8s %8s"
            % ("rate", "contrast", "d_all", "d_worst", "p_worst")
        )
        for rate in args.rates:
            base = load_dnsmos(
                args.dnsmos_dir, corpus, baseline, rate, args.metric
            )
            if base is None:
                continue
            for arm in args.arms[1:]:
                cur = load_dnsmos(
                    args.dnsmos_dir, corpus, arm, rate, args.metric
                )
                if cur is None:
                    continue
                x, y = aligned(base, cur)
                sel = x <= np.percentile(x, 25)
                if sel.sum() < 3:
                    continue
                _, p = stats.ttest_rel(y[sel], x[sel])
                print(
                    "%-6d %-30s %8.3f %8.3f %8.4f"
                    % (
                        rate,
                        "%s - %s" % (arm, baseline),
                        (y - x).mean(),
                        (y - x)[sel].mean(),
                        p,
                    )
                )


def load_decodes(path: str) -> Dict[str, Tuple[float, float, float]]:
    """Read a seedtts decode.txt into {utterance: (wer, errors, ref_words)}."""
    rows: Dict[str, Tuple[float, float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            # seedtts.py appends summary lines that have no tab-separated fields.
            if len(parts) != 7:
                continue
            name = os.path.splitext(os.path.basename(parts[0].replace("\\", "/")))[0]
            wer = float(parts[1])
            errors = sum(float(x) for x in parts[4:7])
            # The reference word count is not written out. Recover it from the
            # rate where possible and fall back to counting the reference text,
            # which is what seedtts.py tokenized.
            words = errors / wer if wer > 0 else float(len(parts[2].split()))
            rows[name] = (wer, errors, words)
    return rows


def run_wer(args) -> None:
    baseline: Optional[str] = args.baseline or args.conditions[0]
    if baseline not in args.conditions:
        raise SystemExit(f"--baseline {baseline} is not in --conditions")

    data = {}
    for cond in args.conditions:
        path = os.path.join(args.results_dir, cond, "decode.txt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}; run zipvoice.eval.wer.seedtts first.")
        data[cond] = load_decodes(path)

    print("-" * 78)
    print(f"{'condition':<34}{'Seed-TTS WER':>14}{'corpus WER':>14}{'n':>6}")
    for cond, rows in data.items():
        seedtts = 100 * float(np.mean([v[0] for v in rows.values()]))
        corpus = 100 * sum(v[1] for v in rows.values()) / sum(
            v[2] for v in rows.values()
        )
        print(f"{cond:<34}{seedtts:>14.2f}{corpus:>14.2f}{len(rows):>6}")

    print("-" * 78)
    print(f"paired against {baseline}, per-utterance WER in percentage points")
    print(f"{'condition':<34}{'delta':>10}{'p (t)':>10}{'p (Wilcoxon)':>14}")
    base = data[baseline]
    for cond, rows in data.items():
        if cond == baseline:
            continue
        keys = sorted(set(rows) & set(base))
        if not keys:
            raise SystemExit(f"No shared utterances between {cond} and {baseline}")
        a = np.array([rows[k][0] for k in keys]) * 100
        b = np.array([base[k][0] for k in keys]) * 100
        _, p_t = stats.ttest_rel(a, b)
        # Wilcoxon is undefined when every pair is tied.
        p_w = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float("nan")
        print(f"{cond:<34}{(a - b).mean():>+10.2f}{p_t:>10.4f}{p_w:>14.4f}")
    print("-" * 78)


def main() -> None:
    args = get_parser().parse_args()
    if args.command == "tail":
        run_tail(args)
    else:
        run_wer(args)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
