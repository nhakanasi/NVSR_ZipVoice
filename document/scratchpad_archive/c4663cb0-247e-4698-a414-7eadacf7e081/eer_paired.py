"""Paired differences in per-speaker EER between conditions, with a speaker-level bootstrap
(10,000 draws, seed 0) and the share of speakers where the first condition is lower.
usage: eer_paired.py SCRATCH A:B [A:B ...]"""
import sys
from pathlib import Path

import numpy as np

rows = [l.rstrip("\n").split("\t") for l in open(Path(sys.argv[1]) / "eer" / "per_speaker.tsv", encoding="utf-8")]
d = {}
for v, c, s, e in rows:
    d.setdefault((v, c), {})[s] = float(e)
rng = np.random.default_rng(0)
for v in sorted({r[0] for r in rows}, reverse=True):
    for pair in sys.argv[2:]:
        a, b = pair.split(":")
        spk = sorted(d[(v, a)])
        x = np.array([d[(v, a)][s] - d[(v, b)][s] for s in spk])
        boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)])
        print(f"{v}\t{a} - {b}\t{x.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
              f"\t{a} lower for {np.mean(x < 0):.0%} of speakers, equal {np.mean(x == 0):.0%}", flush=True)
