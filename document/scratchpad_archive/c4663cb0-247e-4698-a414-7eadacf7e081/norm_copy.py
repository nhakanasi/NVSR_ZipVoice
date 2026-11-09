"""Loudness-normalised copies of every listening-test condition, for rescoring both verifiers on
what listeners hear. Same processing as the pilot bundle (build_pilot.py): -23 LUFS integrated,
turned down to a 0.99 peak if needed, written as 16-bit WAV. All 76 LibriTTS items, so the
numbers compare with the unnormalised scores; the pilot's 24 items are a subset.
Also writes norm/test_score.tsv, the eval list with its reference column pointing at the
normalised clean prompts.
usage: norm_copy.py SCRATCH"""
import sys
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf

S = sys.argv[1]
RS = "C:/zipvoice_data/exp/results_rs/libritts"
SRC = {
    "ref": "C:/zipvoice_data/data/bwe_eval/prompts/clean",
    "nat24": f"{S}/control1/libritts/sr24000",
    "nat8": f"{S}/control1/libritts/sr8000",
    "audiosr": f"{S}/control2/libritts/audiosr_sr8000_trim",
    "nvsr": f"{S}/control2/libritts/nvsr_sr8000_fit",
    "stock24": f"{RS}/stock_sr24000",
    "stock8": f"{RS}/stock_sr8000",
    "cascade": f"{RS}/cascade_nvsr_sr8000",
}
LUFS, SR = -23.0, 24000
meter = pyloudnorm.Meter(SR)
lines = [l.rstrip("\n").split("\t") for l in open("C:/zipvoice_data/data/bwe_eval/test_score.tsv", encoding="utf-8")]
uids = [l[0] for l in lines]
for system, src in SRC.items():
    out = Path(S) / "norm" / system
    out.mkdir(parents=True, exist_ok=True)
    gains = []
    for u in uids:
        w, sr = sf.read(f"{src}/{u}.wav", dtype="float64")
        assert sr == SR and w.ndim == 1, (system, u, sr, w.shape)
        before = meter.integrated_loudness(w)
        y = pyloudnorm.normalize.loudness(w, before, LUFS)
        peak = np.abs(y).max()
        if peak > 0.99:
            y *= 0.99 / peak
        sf.write(out / f"{u}.wav", y, SR, subtype="PCM_16")
        gains.append(20 * np.log10(np.abs(y).max() / np.abs(w).max()))
    print(f"{system:8} gain applied: median {np.median(gains):+.1f} dB, range {min(gains):+.1f} to {max(gains):+.1f}")
with open(Path(S) / "norm" / "test_score.tsv", "w", encoding="utf-8", newline="\n") as f:
    for l in lines:
        l[2] = f"{S}/norm/ref/{l[0]}.wav"
        f.write("\t".join(l) + "\n")
