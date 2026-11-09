"""Band-matched scoring, step 3: does scoring both sides in the prompt's band remove the bandwidth effect?

Published (full) scoring: SIM(x) = cos(E(x), E(clean prompt)).
Band-matched: SIM_m(x) = cos(E(LP x), E(LP prompt)), LP = 24k -> 8k -> 24k (bandmatch_prep.py).
Proposal under test: score the 8 kHz condition band-matched, the 24 kHz condition as published.
  B_m = n24 - n8m      bandwidth effect left on genuine speech (0 if the fix works)
  D_m = s24 - s8m      a system's drop under the fix
  I_m = D_m - B_m      residual under the fix, compared with the published I = D - B
Also the low band alone (both conditions band-matched) and the Table 3 contrasts under the fix.
Speaker-level bootstrap (LibriTTS: 38 speakers x 2 items; VCTK: 76 x 1), 10,000 draws, seed 0."""
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
OS = "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
V = {"W": "sim_scores.tsv", "C": "campp_scores.tsv"}
N = 10000


def load(path):
    return {u: float(v) for u, v in (l.rstrip("\n").split("\t") for l in list(open(path, encoding="utf-8"))[1:])}


for corpus in ("libritts", "vctk"):
    rs, c2 = f"C:/zipvoice_data/exp/results_rs/{corpus}", f"{OS}/control2/{corpus}"
    full = {"n24": f"{OS}/control1/{corpus}/sr24000", "n8": f"{OS}/control1/{corpus}/sr8000",
            "z24": f"{rs}/stock_sr24000", "z8": f"{rs}/stock_sr8000", "f24": f"{rs}/f5_sr24000",
            "f8": f"{rs}/f5_sr8000", "cas": f"{rs}/cascade_nvsr_sr8000",
            "asr": f"{c2}/audiosr_sr8000_trim", "nvsr": f"{c2}/nvsr_sr8000_fit"}
    dirs = {**full, **{k + "m": str(SP / "bandmatch" / corpus / k) for k in full}}
    uids = sorted(load(f"{dirs['n24m']}/{V['W']}"))
    spk = np.array([u.split("_")[0] for u in uids])
    a = {(k, v): np.array([load(f"{d}/{f}")[u] for u in uids]) for k, d in dirs.items() for v, f in V.items()}
    groups = [np.flatnonzero(spk == s) for s in np.unique(spk)]
    rng = np.random.default_rng(0)
    idx = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(N)]
    print(f"\n##### {corpus}: {len(uids)} items, {len(groups)} speakers")
    for v in V:
        x = lambda k: a[(k, v)]

        def row(name, f):
            b = np.array([f(i) for i in idx])
            print(f"  {v} {name:46s} {f(np.arange(len(uids))):+.3f} "
                  f"[{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")

        d = lambda p, q: (lambda i: (x(p)[i] - x(q)[i]).mean())
        print(f"  {v} means: " + " ".join(f"{k}={x(k).mean():.3f}" for k in dirs))
        row("B   = n24 - n8   (published)", d("n24", "n8"))
        row("B_m = n24 - n8m  (fix; 0 if it works)", d("n24", "n8m"))
        row("n8m - n8  (what matching the prompt buys)", d("n8m", "n8"))
        for s in ("z", "f"):
            I = lambda i, s=s: (x("n8")[i] - x(s + "8")[i] - x("n24")[i] + x(s + "24")[i]).mean()
            Im = lambda i, s=s: (x("n8m")[i] - x(s + "8m")[i] - x("n24")[i] + x(s + "24")[i]).mean()
            row(f"{s}: D   = {s}24 - {s}8", d(s + "24", s + "8"))
            row(f"{s}: D_m = {s}24 - {s}8m", d(s + "24", s + "8m"))
            row(f"{s}: I   (published)", I)
            row(f"{s}: I_m (fix)", Im)
            row(f"{s}: I_m - I", lambda i: Im(i) - I(i))
            row(f"{s}: low band only, {s}24m - {s}8m", d(s + "24m", s + "8m"))
        row("genuine low band only, n24m - n8m", d("n24m", "n8m"))
        row("cascade - ZipVoice at 8k (published)", d("cas", "z8"))
        row("cascade - ZipVoice at 8k (fix)", d("casm", "z8m"))
        for k in ("asr", "nvsr"):
            row(f"{k} - n8 (published)", d(k, "n8"))
            row(f"{k} - n8 (fix, both low-passed)", d(k + "m", "n8m"))
