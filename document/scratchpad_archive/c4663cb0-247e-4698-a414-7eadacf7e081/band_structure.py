"""Upper-band structure per system: spectral flatness (1 = white noise, low =
peaky/formant/harmonic) and spectral flux (how much the band's shape changes
frame to frame), over speech frames, in 4-8 kHz and 8-12 kHz."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SR = 24000
BANDS = [(4000, 8000), (8000, 12000)]


def stats(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2 + 1e-12
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    low = S[(f >= 100) & (f < 2000)].sum(0)
    ldb = 10 * np.log10(low)
    speech = ldb > ldb.max() - 35
    out = []
    for lo, hi in BANDS:
        B = S[(f >= lo) & (f < hi)][:, speech]
        flat = np.exp(np.log(B).mean(0)) / B.mean(0)
        L = np.log(B)
        L = L - L.mean(0, keepdims=True)  # shape only, not level
        flux = np.abs(np.diff(L, axis=1)).mean()
        out += [flat.mean(), flux]
    return out


print(f"{'system':18s} flat4-8 flux4-8 flat8-12 flux8-12")
for arg in sys.argv[1:]:
    d, name = arg.split("=")
    rows = [stats(p) for p in sorted(Path(d).glob("*.wav"))]
    m = np.mean(rows, axis=0)
    print(f"{name:18s} " + " ".join(f"{v:7.3f}" for v in m) + f"  n={len(rows)}")
