"""Summarise the per-sentence RTF that infer_zipvoice logs.

The first sentence of every run is discarded: it carries CUDA context setup
and kernel autotuning that no later sentence pays, and including it would
flatter whichever system happened to run last.
"""

import argparse
import io
import re
from collections import defaultdict

DIR_RE = re.compile(r"[/\]([A-Za-z_]+)_sr(\d+)[/\]")
RTF_RE = re.compile(r"\[Sentence: (\d+)\] RTF: ([\d.]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="inference log files to parse")
    args = parser.parse_args()

    runs = defaultdict(list)
    for path in args.logs:
        current = None
        with io.open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                found = DIR_RE.search(line)
                if found:
                    current = (found.group(1), int(found.group(2)))
                found = RTF_RE.search(line)
                if found and current is not None:
                    index, value = int(found.group(1)), float(found.group(2))
                    if index > 0:
                        runs[current].append(value)

    print(f"{'system':20s} {'rate':>6s} {'n':>4s} {'mean':>8s} "
          f"{'median':>8s} {'min':>7s} {'max':>7s}")
    for (system, rate) in sorted(runs):
        v = sorted(runs[(system, rate)])
        mid = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1]
                                                 + v[len(v) // 2]) / 2
        print(f"{system:20s} {rate:6d} {len(v):4d} {sum(v) / len(v):8.4f} "
              f"{mid:8.4f} {v[0]:7.4f} {v[-1]:7.4f}")


if __name__ == "__main__":
    main()
