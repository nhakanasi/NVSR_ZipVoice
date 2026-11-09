import sys, glob, os, numpy as np, soundfile as sf
from scipy.signal import resample_poly, stft
def bands(p):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim == 2: a = a.mean(1)
    if sr != 24000:
        from math import gcd; g = gcd(sr, 24000); a = resample_poly(a, 24000//g, sr//g)
    f, t, s = stft(a, fs=24000, nperseg=1024, noverlap=768, boundary=None)
    p = np.abs(s)**2
    tot = p.sum()
    return [10*np.log10(p[(f>=lo)&(f<hi)].sum()/tot + 1e-12) for lo, hi in [(4e3,6e3),(6e3,8e3),(8e3,12e3)]]
ref = "C:/zipvoice_data/data/bwe_eval/references"
R = "C:/zipvoice_data/exp/results"
for name in sys.argv[1:]:
    d = ref if name == "ref" else f"{R}/{name}"
    v = np.array([bands(p) for p in sorted(glob.glob(d + "/*.wav"))])
    print(f"{name:32s} n={len(v):3d}  4-6k {v[:,0].mean():6.1f}  6-8k {v[:,1].mean():6.1f}  8-12k {v[:,2].mean():6.1f} dB rel total")
