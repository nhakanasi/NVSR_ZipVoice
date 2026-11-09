"""Mean band levels (dB) per system over the eval set, and per-utterance SIM deltas."""
import math, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, stft

R = Path("C:/zipvoice_data/exp/results")
REF = Path("C:/zipvoice_data/data/bwe_eval/references")


def spec_db(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim == 2:
        a = a.mean(1)
    if sr != 24000:
        g = math.gcd(sr, 24000)
        a = resample_poly(a, 24000 // g, sr // g)
    f, _, s = stft(a, fs=24000, nperseg=1024, noverlap=768, boundary=None)
    p = np.abs(s) ** 2
    # Speech frames only: top 70% by energy, so silence length does not set the level.
    e = p.sum(0)
    keep = e >= np.quantile(e, 0.3)
    return f, p[:, keep].mean(1)


def bands(f, p, cut):
    out = {}
    for name, lo, hi in [("low", 100, cut), ("hi1", cut, min(cut * 2, 12000)), ("hi2", min(cut * 2, 12000), 12000)]:
        if hi > lo:
            out[name] = 10 * np.log10(p[(f >= lo) & (f < hi)].mean() + 1e-12)
    return out


for rate in ([int(r) for r in sys.argv[1:]] if __name__ == "__main__" else []):
    cut = rate // 2
    systems = {"ref": None}
    for k in ("10k", "steermasked10k", "hbfix2k"):
        d = R / (f"adapter_{k}_sr{rate}" if k.startswith(("hbfix", "steermasked")) else f"adapter_steer{k}_sr{rate}")
        if d.exists():
            systems[f"steer{k}"] = d
    names = sorted(p.stem for p in systems["steer10k"].glob("*.wav"))
    acc = {k: [] for k in systems}
    for n in names:
        for k, d in systems.items():
            path = REF / f"{n}.wav" if d is None else d / f"{n}.wav"
            if path.exists():
                acc[k].append(bands(*spec_db(path), cut))
    print(f"sr{rate} (cutoff {cut} Hz), mean dB over {len(names)} utts, speech frames")
    for k, rows in acc.items():
        if rows:
            print(f"  {k:9s}", "  ".join(f"{b} {np.mean([r[b] for r in rows]):6.1f}" for b in rows[0]))
