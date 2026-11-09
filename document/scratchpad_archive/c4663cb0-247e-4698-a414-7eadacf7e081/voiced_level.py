"""High-band level on voiced frames vs weaker speech frames, relative to each
frame's own 100-2000 Hz level (dB), averaged over files."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa

SR = 24000
def stats(p):
    x, sr = sf.read(p)
    if x.ndim > 1: x = x.mean(1)
    if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2 + 1e-12
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    db = lambda lo, hi: 10 * np.log10(S[(f >= lo) & (f < hi)].sum(0))
    low = db(100, 2000); b1 = db(4000, 8000); b2 = db(8000, 12000)
    voiced = low > low.max() - 15
    weak = (low <= low.max() - 15) & (low > low.max() - 35)
    return [np.mean(b1[voiced] - low[voiced]), np.mean(b2[voiced] - low[voiced]),
            np.mean(b1[weak] - low[weak]), np.mean(b2[weak] - low[weak])]
print(f"{'system':14s} voiced:4-8 8-12  weak:4-8 8-12   (dB re 100-2000 Hz)")
for a in sys.argv[1:]:
    d, n = a.split("=")
    m = np.nanmean([stats(p) for p in sorted(Path(d).glob("*.wav"))], 0)
    print(f"{n:14s} " + "  ".join(f"{v:6.1f}" for v in m))
