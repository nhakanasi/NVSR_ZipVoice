"""Both verifiers on the original vs the -23 LUFS normalised listening-test conditions (review 15,
point 1). Per verifier: condition means; the six contrasts (A - B) on all 76 items and on the
pilot's 24; per session, stock24 - nat24, the CAM++ - SIM-o offset and identity lost
((stock24 - nat24) - (stock8 - nat8)). 95% intervals from 10,000 item bootstraps within session.
usage: compare_norm.py SCRATCH BUNDLE_DIR"""
import json
import sys

import numpy as np

S, BUNDLE = sys.argv[1], sys.argv[2]
RS = "C:/zipvoice_data/exp/results_rs/libritts"
ORIG = {"nat24": f"{S}/control1/libritts/sr24000", "nat8": f"{S}/control1/libritts/sr8000",
        "audiosr": f"{S}/control2/libritts/audiosr_sr8000_trim", "nvsr": f"{S}/control2/libritts/nvsr_sr8000_fit",
        "stock24": f"{RS}/stock_sr24000", "stock8": f"{RS}/stock_sr8000", "cascade": f"{RS}/cascade_nvsr_sr8000"}
CON = {"C1": ("nat24", "nat8"), "C2": ("audiosr", "nat8"), "C3": ("nvsr", "nat8"),
       "C4": ("stock8", "nat8"), "C5": ("cascade", "stock8"), "C6": ("stock24", "nat24")}
V = {"SIM-o": "sim_scores.tsv", "CAM++": "campp_scores.tsv"}
grp = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in open(f"{S}/libritts_chapter.tsv", encoding="utf-8")}
uids = sorted(grp)
P = json.loads(open(f"{BUNDLE}/trials.js", encoding="utf-8").read().split("=", 1)[1].rstrip().rstrip(";"))
pilot = sorted(u for b in P["blocks"] for u in b)
rng = np.random.default_rng(0)


def load(path):
    return {u: float(v) for u, v in (l.rstrip("\n").split("\t") for l in list(open(path, encoding="utf-8"))[1:])}


x = {(set_, v, c): load(f"{ORIG[c] if set_ == 'orig' else S + '/norm/' + c}/{f}")
     for set_ in ["orig", "norm"] for v, f in V.items() for c in ORIG}


def arr(set_, v, c, us):
    return np.array([x[set_, v, c][u] for u in us])


def ci(vals, us):
    """Mean and 95% bootstrap interval, resampling within session so the 38/38 mix is kept."""
    g = np.array([grp[u] for u in us])
    parts = [np.flatnonzero(g == k) for k in ["same", "diff"] if (g == k).any()]
    b = [vals[np.concatenate([p[rng.integers(0, len(p), len(p))] for p in parts])].mean() for _ in range(10000)]
    return f"{vals.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"


print("condition means (orig -> norm)")
for v in V:
    print(f"  {v:6} " + "  ".join(f"{c} {arr('orig', v, c, uids).mean():.3f}->{arr('norm', v, c, uids).mean():.3f}"
                                  for c in ORIG))
for name, us in [("all 76 items", uids), ("pilot's 24 items", pilot)]:
    print(f"\ncontrasts A - B, {name}")
    for cid, (A, B) in CON.items():
        print(f"  {cid} {A:>7} - {B:<7} " + "   ".join(
            f"{v} {s} {ci(arr(s, v, A, us) - arr(s, v, B, us), us)}" for v in V for s in ["orig", "norm"]))
print("\nper session (all 76)")
for g in ["same", "diff"]:
    us = [u for u in uids if grp[u] == g]
    for s in ["orig", "norm"]:
        bonus = {v: arr(s, v, "stock24", us) - arr(s, v, "nat24", us) for v in V}
        il = {v: bonus[v] - (arr(s, v, "stock8", us) - arr(s, v, "nat8", us)) for v in V}
        print(f"  {g:4} {s}: stock24 - nat24 SIM-o {ci(bonus['SIM-o'], us)} CAM++ {ci(bonus['CAM++'], us)}; "
              f"offset {ci(bonus['CAM++'] - bonus['SIM-o'], us)}")
        print(f"            identity lost SIM-o {ci(il['SIM-o'], us)} CAM++ {ci(il['CAM++'], us)}")
for s in ["orig", "norm"]:
    il = {v: (arr(s, v, "stock24", uids) - arr(s, v, "nat24", uids))
          - (arr(s, v, "stock8", uids) - arr(s, v, "nat8", uids)) for v in V}
    off = (arr(s, "CAM++", "stock24", uids) - arr(s, "CAM++", "nat24", uids)) \
        - (arr(s, "SIM-o", "stock24", uids) - arr(s, "SIM-o", "nat24", uids))
    print(f"  all  {s}: identity lost SIM-o {ci(il['SIM-o'], uids)} CAM++ {ci(il['CAM++'], uids)}; offset {ci(off, uids)}")
print("\nper-item change from normalisation, |norm - orig| median / max")
for v in V:
    d = np.concatenate([np.abs(arr("norm", v, c, uids) - arr("orig", v, c, uids)) for c in ORIG])
    print(f"  {v}: {np.median(d):.4f} / {d.max():.4f}")
