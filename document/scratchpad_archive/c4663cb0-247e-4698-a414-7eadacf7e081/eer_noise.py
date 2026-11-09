"""White-noise conditions for the EER companion, on the same perturbed halves as eer_prep.py:
Gaussian white noise at 40, 20 and 0 dB SNR against each file's own power (seed 0), matching
Carbonneau et al.'s noise conditions.
usage: eer_noise.py SCRATCH"""
import sys
from pathlib import Path

import torch
import torchaudio

S = Path(sys.argv[1]) / "eer"
g = torch.Generator().manual_seed(0)
for snr in [40, 20, 0]:
    out = S / f"noise{snr}"
    out.mkdir(exist_ok=True)
    for f in sorted((S / "clean").glob("*.wav")):
        w, sr = torchaudio.load(str(f))
        n = torch.randn(w.shape, generator=g)
        n *= (w.pow(2).mean() / n.pow(2).mean() / 10 ** (snr / 10)).sqrt()
        torchaudio.save(str(out / f.name), w + n, sr)
