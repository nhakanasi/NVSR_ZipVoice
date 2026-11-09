"""Channel-copy test on LibriTTS: is stock's full-band lead over natural speech (stock24 -
nat24) larger when the natural reference comes from a different chapter (recording session)
than the prompt? Bootstrap within each group; the gap between groups by independent draws.
usage: chapter_test.py SCRATCH"""
import sys

import numpy as np

S = sys.argv[1]
RS = "C:/zipvoice_data/exp/results_rs/libritts"
rng = np.random.default_rng(0)
grp = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in open(f"{S}/libritts_chapter.tsv", encoding="utf-8")}


def load(path):
    return {u: float(v) for u, v in (l.rstrip("\n").split("\t") for l in list(open(path, encoding="utf-8"))[1:])}


def boot(x, n=10000):
    return np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])


dirs = {"nat24": f"{S}/control1/libritts/sr24000", "nat8": f"{S}/control1/libritts/sr8000",
        "stock24": f"{RS}/stock_sr24000", "stock8": f"{RS}/stock_sr8000"}
for score in ["sim_scores.tsv", "campp_scores.tsv"]:
    d = {k: load(f"{v}/{score}") for k, v in dirs.items()}
    print(f"== {score}")
    res = {}
    for g in ["same", "diff"]:
        u = [x for x in grp if grp[x] == g]
        a = {k: np.array([d[k][x] for x in u]) for k in d}
        bonus = a["stock24"] - a["nat24"]
        lost = (a["nat8"] - a["stock8"]) - (a["nat24"] - a["stock24"])
        res[g] = (boot(bonus), boot(lost))
        print(f"  {g} chapter n={len(u)}: nat24 {a['nat24'].mean():.3f} stock24 {a['stock24'].mean():.3f} "
              f"bonus {bonus.mean():+.3f} [{np.percentile(res[g][0], 2.5):+.3f}, {np.percentile(res[g][0], 97.5):+.3f}] "
              f"identity lost {lost.mean():+.3f} [{np.percentile(res[g][1], 2.5):+.3f}, {np.percentile(res[g][1], 97.5):+.3f}]")
    gap = res["diff"][0] - res["same"][0]
    print(f"  bonus diff - same {np.mean(gap):+.3f} [{np.percentile(gap, 2.5):+.3f}, {np.percentile(gap, 97.5):+.3f}]")
