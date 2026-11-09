"""4-8k and 8-12k level on speech frames two ways: power mean (what the band
metrics and the ear see) and log mean (what a log-mel loss matches). dB re each
frame's 100-2000 Hz power."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
SR = 24000
def stats(p):
    x, sr = sf.read(p)
    if x.ndim > 1: x = x.mean(1)
    if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2 + 1e-12
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=100, fmin=0, fmax=12000)
    mf = librosa.mel_frequencies(n_mels=100, fmin=0, fmax=12000)
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    low = 10 * np.log10(S[(f >= 100) & (f < 2000)].mean(0))
    sp = low > low.max() - 35
    out = []
    for lo, hi in [(4000, 8000), (8000, 12000)]:
        M = mel[(mf >= lo) & (mf < hi)][:, sp] + 1e-12
        out += [np.mean(10 * np.log10(M.mean(0)) - low[sp]),
                np.mean(10 * np.log10(M).mean(0) - low[sp])]
    return out
print(f"{'system':12s} 4-8k:power  log  gap   8-12k:power  log  gap")
for a in sys.argv[1:]:
    d, n = a.split("=")
    m = np.mean([stats(p) for p in sorted(Path(d).glob("*.wav"))], 0)
    print(f"{n:12s} {m[0]:10.1f} {m[1]:5.1f} {m[0]-m[1]:4.1f} {m[2]:12.1f} {m[3]:5.1f} {m[2]-m[3]:4.1f}")
