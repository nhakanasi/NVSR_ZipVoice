"""Why only CAM++ rates stock above natural at 24 kHz, when both verifiers show the same session
bonus. Per chapter group (same / different session as the prompt): stock24 - nat24 under each
verifier, the CAM++ minus SIM-o offset (paired within items), and the session-mix at which each
verifier's pooled bonus would cross zero. Bootstrap 10,000 draws within groups.
usage: chapter_offset.py SCRATCH"""
import sys

import numpy as np

S = sys.argv[1]
RS = "C:/zipvoice_data/exp/results_rs/libritts"
rng = np.random.default_rng(0)
grp = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in open(f"{S}/libritts_chapter.tsv", encoding="utf-8")}


def load(path):
    return {u: float(v) for u, v in (l.rstrip("\n").split("\t") for l in list(open(path, encoding="utf-8"))[1:])}


def ci(b):
    return f"[{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"


bonus = {}
for v, score in [("SIM-o", "sim_scores.tsv"), ("CAM++", "campp_scores.tsv")]:
    nat, stock = load(f"{S}/control1/libritts/sr24000/{score}"), load(f"{RS}/stock_sr24000/{score}")
    bonus[v] = {g: np.array([stock[u] - nat[u] for u in sorted(grp) if grp[u] == g]) for g in ["same", "diff"]}
n = {g: len(bonus["SIM-o"][g]) for g in ["same", "diff"]}
print(f"same-chapter share of the 76 references: {n['same']}/{n['same'] + n['diff']} "
      f"= {n['same'] / (n['same'] + n['diff']):.2f}")
idx = {g: [rng.integers(0, n[g], n[g]) for _ in range(10000)] for g in n}
for g in ["same", "diff"]:
    off = bonus["CAM++"][g] - bonus["SIM-o"][g]
    print(f"{g}: stock - natural  SIM-o {bonus['SIM-o'][g].mean():+.3f}  CAM++ {bonus['CAM++'][g].mean():+.3f}  "
          f"CAM++ - SIM-o offset {off.mean():+.3f} {ci(np.array([off[i].mean() for i in idx[g]]))}")
inter = np.array([(bonus["CAM++"]["diff"][i] - bonus["SIM-o"]["diff"][i]).mean()
                  - (bonus["CAM++"]["same"][j] - bonus["SIM-o"]["same"][j]).mean()
                  for i, j in zip(idx["diff"], idx["same"])])
print(f"offset, different - same (verifier x session interaction): {inter.mean():+.3f} {ci(inter)}")
for v in ["SIM-o", "CAM++"]:
    s, d = bonus[v]["same"].mean(), bonus[v]["diff"].mean()
    pooled = (n["same"] * s + n["diff"] * d) / (n["same"] + n["diff"])
    cross = d / (d - s) if (d > 0) != (s > 0) else None
    print(f"{v}: pooled bonus {pooled:+.3f}; zero crossing at same-chapter share "
          + (f"{cross:.2f}" if cross is not None else "none (same sign in both groups)"))
