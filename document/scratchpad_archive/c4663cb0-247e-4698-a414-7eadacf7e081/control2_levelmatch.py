"""Control 2, level-matched: is AudioSR's SIM-o penalty the band's level or its content?

Per utterance, the added band (4.2-12 kHz) of the trimmed AudioSR output is rescaled in the
STFT domain to the natural 24 kHz reference's energy in the same range. The reference is the
same utterance, so this is an oracle level, stronger than any practical re-equalisation.
Two variants:
  gain  one gain over the whole 4.2-12 kHz range
  eq16  16 equal-width bands over 4.2-12 kHz, each matched separately; this is the
        bandwidth analogue of Carbonneau et al.'s 16-band re-equalisation, restricted to
        the added band so the genuine low band is untouched
usage: control2_levelmatch.py SCRATCH [SOURCE_VARIANT [OUT_PREFIX]]
  defaults: audiosr_sr8000_trim, audiosr_lm"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

S = sys.argv[1]
SRC = sys.argv[2] if len(sys.argv) > 2 else "audiosr_sr8000_trim"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "audiosr_lm"
N_FFT, HOP, LO, HI = 2048, 512, 4200, 12000
win = torch.hann_window(N_FFT)
f = torch.fft.rfftfreq(N_FFT, 1 / 24000)
band = (f > LO) & (f <= HI)
edges = np.linspace(LO, HI, 17)
sub = [(f > a) & (f <= b) for a, b in zip(edges[:-1], edges[1:])]


def stft(w):
    return torch.stft(w, N_FFT, HOP, window=win, return_complex=True)


for corpus in ["libritts", "vctk"]:
    src = Path(f"{S}/control2/{corpus}/{SRC}")
    outs = {v: Path(f"{S}/control2/{corpus}/{PREFIX}_{v}") for v in ["gain", "eq16"]}
    for o in outs.values():
        o.mkdir(exist_ok=True)
    g_db = []
    for p in sorted(src.glob("*.wav")):
        w, sr = torchaudio.load(str(p))
        r, _ = torchaudio.load(f"{S}/control1/{corpus}/sr24000/{p.name}")
        assert sr == 24000 and w.shape == r.shape, (p.name, w.shape, r.shape)
        X, P = stft(w[0]), stft(r[0]).abs().pow(2).sum(1)
        E = X.abs().pow(2).sum(1)
        for v, masks in {"gain": [band], "eq16": sub}.items():
            gain = torch.ones_like(f)
            for m in masks:
                gain[m] = (P[m].sum() / E[m].sum().clamp_min(1e-12)).sqrt()
            if v == "gain":
                g_db.append(20 * np.log10(gain[band][0].item()))
            y = torch.istft(X * gain[:, None], N_FFT, HOP, window=win, length=w.size(1))
            torchaudio.save(str(outs[v] / p.name), y[None], sr)
    print(corpus, f"single gain applied, dB: median {np.median(g_db):.1f} "
          f"range [{min(g_db):.1f}, {max(g_db):.1f}]", flush=True)
