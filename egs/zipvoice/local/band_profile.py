import sys, glob, os
import numpy as np
import soundfile as sf

BANDS = [(100,2000),(2000,3000),(3000,4000),(4000,6000),(6000,8000),(8000,12000)]

def profile(path):
    x, sr = sf.read(path)
    if x.ndim > 1: x = x.mean(1)
    n = 1 << 15
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    hop, win = n//2, np.hanning(n)
    fr = [np.abs(np.fft.rfft(x[i:i+n]*win))**2 for i in range(0, len(x)-n+1, hop)]
    if not fr: return None
    P = np.mean(fr, 0); f = np.fft.rfftfreq(n, 1/sr)
    return np.array([P[(f>=a)&(f<b)].sum() for a,b in BANDS])

print(f"{'system':26s}" + "".join(f"{a//1000}-{b//1000}k".rjust(9) for a,b in BANDS))
for arg in sys.argv[1:]:
    # ``path`` or ``path=label``; the label is what the row is named.
    d, _, label = arg.partition("=")
    label = label or os.path.basename(d.rstrip("/\\"))
    acc = []
    for w in sorted(glob.glob(os.path.join(d,"*.wav"))):
        p = profile(w)
        if p is not None: acc.append(p)
    if not acc:
        print(f"{label:26s}   no .wav files under {d}")
        continue
    A = np.mean(acc,0)
    # every band in dB relative to that system's own 100-2000 Hz band
    rel = 10*np.log10((A + 1e-20)/(A[0] + 1e-20))
    print(f"{label:26s}" + "".join(f"{v:9.1f}" for v in rel) + f"   n={len(acc)}")
