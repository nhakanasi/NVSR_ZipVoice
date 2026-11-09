"""Upper-band structure sweep. Per system, over speech frames:
lvl    power level of 4-8k / 8-12k relative to 100-2000 Hz (dB)
flat   STFT spectral flatness (1 = white)
flux   frame-to-frame change of the band's log-spectral shape
ctr    log-mel envelope contrast: per-frame std across mel bins of the band
       after removing a linear tilt (formant/peak structure the mel can carry)
dyn    std over time of the band's log energy (how much the band moves)
corr   correlation over time of band log energy with low-band log energy
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SR = 24000
BANDS = [(4000, 8000), (8000, 12000)]
MEL = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=100, fmin=0, fmax=12000)
MELF = librosa.mel_frequencies(n_mels=102, fmin=0, fmax=12000)[1:-1]


def stats(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2 + 1e-12
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    M = np.log(MEL @ np.sqrt(S) + 1e-7)
    low = S[(f >= 100) & (f < 2000)].sum(0)
    ldb = 10 * np.log10(low)
    sp = ldb > ldb.max() - 35
    out = []
    for lo, hi in BANDS:
        B = S[(f >= lo) & (f < hi)][:, sp]
        lvl = 10 * np.log10(B.sum(0).mean() / low[sp].mean())
        flat = (np.exp(np.log(B).mean(0)) / B.mean(0)).mean()
        L = np.log(B)
        L = L - L.mean(0, keepdims=True)
        flux = np.abs(np.diff(L, axis=1)).mean()
        mb = (MELF >= lo) & (MELF < hi)
        Mb = M[mb][:, sp]
        i = np.arange(Mb.shape[0])
        A = np.stack([i, np.ones_like(i)], 1)
        coef, *_ = np.linalg.lstsq(A, Mb, rcond=None)
        ctr = (Mb - A @ coef).std(0).mean()
        e = 10 * np.log10(B.sum(0))
        dyn = e.std()
        corr = np.corrcoef(e, ldb[sp])[0, 1]
        out += [lvl, flat, flux, ctr, dyn, corr]
    return out


hdr = ["lvl", "flat", "flux", "ctr", "dyn", "corr"]
print(f"{'system':22s} " + " ".join(f"{h+'4':>7s}" for h in hdr) + " | "
      + " ".join(f"{h+'8':>7s}" for h in hdr))
for arg in sys.argv[1:]:
    d, name = arg.split("=")
    files = sorted(Path(d).glob("*.wav"))
    rows = [stats(p) for p in files]
    m = np.mean(rows, axis=0)
    print(f"{name:22s} " + " ".join(f"{v:7.3f}" for v in m[:6]) + " | "
          + " ".join(f"{v:7.3f}" for v in m[6:]) + f"  n={len(rows)}", flush=True)
