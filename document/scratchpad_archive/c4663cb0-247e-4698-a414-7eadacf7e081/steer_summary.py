"""Scores per steer checkpoint from the eval logs, and paired SIM change against a base system."""
import re, sys
from pathlib import Path
import numpy as np

EXP = Path("C:/zipvoice_data/exp")
R = EXP / "results"
LOGS = {"adapter_steer10k": "steer_eval_10000.log", "adapter_steer13k": "steer_eval_13000.log"}
base, systems = sys.argv[1], sys.argv[2:]


def log_scores(system):
    log = EXP / LOGS.get(system, f"eval_{system}.log")
    out, rate = {}, None
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(rf"{system}_sr(\d+)", line)
        if m:
            rate = int(m.group(1))
        for key, pat in [("sim", r"SIM-o score: ([\d.]+)"), ("wer", r"\] WER: ([\d.]+)"),
                         ("utmos", r"UTMOS score: ([\d.]+)"), ("ovrl", r"DNSMOS OVRL: ([\d.]+)"),
                         ("p808", r"DNSMOS P808_MOS: ([\d.]+)")]:
            m = re.search(pat, line)
            if m and rate:
                out.setdefault(rate, {})[key] = float(m.group(1))
    return out


def sims(system, rate):
    rows = (R / f"{system}_sr{rate}" / "sim_scores.tsv").read_text().splitlines()[1:]
    return dict((a, float(b)) for a, b in (r.split("\t") for r in rows if "\t" in r))


scores = {s: log_scores(s) for s in [base] + systems}
for rate in (4000, 8000, 16000):
    print(f"sr{rate}")
    b = sims(base, rate)
    for s in [base] + systems:
        v = scores[s].get(rate, {})
        line = f"  {s:22s} " + "  ".join(f"{k} {v.get(k, float('nan')):.3f}" for k in ("sim", "wer", "utmos", "ovrl", "p808"))
        if s != base:
            c = sims(s, rate)
            d = np.array([c[k] - b[k] for k in b if k in c])
            line += f"   dSIM vs {base} {d.mean():+.3f} (se {d.std(ddof=1) / np.sqrt(len(d)):.4f}, worse {int((d < 0).sum())}/{len(d)})"
        print(line)
