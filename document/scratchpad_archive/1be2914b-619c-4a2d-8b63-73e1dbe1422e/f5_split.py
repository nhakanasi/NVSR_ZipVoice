"""Eq. 1 split for stock F5-TTS beside stock ZipVoice, both verifiers, both corpora.

Same pairing and bootstrap as the ZipVoice rows (decompose.py): utterances resampled
once per draw, 10,000 draws, seed 0, shared by every system so B is common."""
import numpy as np

S = "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
RS = "C:/zipvoice_data/exp/results_rs"
SYSTEMS = {"n24": S + "/control1/{c}/sr24000", "n8": S + "/control1/{c}/sr8000",
           "z24": RS + "/{c}/stock_sr24000", "z8": RS + "/{c}/stock_sr8000",
           "f24": RS + "/{c}/f5_sr24000", "f8": RS + "/{c}/f5_sr8000"}


def load(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")][1:]
    return {u: float(v) for u, v in rows}


for corpus in ["libritts", "vctk"]:
    for score in ["sim_scores.tsv", "campp_scores.tsv"]:
        rng = np.random.default_rng(0)
        d = {k: load(v.format(c=corpus) + "/" + score) for k, v in SYSTEMS.items()}
        uids = sorted(d["n24"])
        assert all(sorted(x) == uids for x in d.values()), "unpaired"
        a = {k: np.array([x[u] for u in uids]) for k, x in d.items()}
        idx = [rng.integers(0, len(uids), len(uids)) for _ in range(10000)]
        stats = {
            "B": lambda i: a["n24"][i].mean() - a["n8"][i].mean(),
            "D_zip": lambda i: a["z24"][i].mean() - a["z8"][i].mean(),
            "I_zip": lambda i: (a["n8"][i] - a["z8"][i] - a["n24"][i] + a["z24"][i]).mean(),
            "D_f5": lambda i: a["f24"][i].mean() - a["f8"][i].mean(),
            "I_f5": lambda i: (a["n8"][i] - a["f8"][i] - a["n24"][i] + a["f24"][i]).mean(),
            "B/D_f5": lambda i: (a["n24"][i].mean() - a["n8"][i].mean()) / (a["f24"][i].mean() - a["f8"][i].mean()),
            "I_f5-I_zip": lambda i: (a["z8"][i] - a["f8"][i] - a["z24"][i] + a["f24"][i]).mean(),
        }
        full = np.arange(len(uids))
        print(f"== {corpus} {score} n={len(uids)} means " + " ".join(f"{k} {v.mean():.4f}" for k, v in a.items()))
        for name, f in stats.items():
            b = np.array([f(i) for i in idx])
            print(f"  {name:12s} {f(full):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")
