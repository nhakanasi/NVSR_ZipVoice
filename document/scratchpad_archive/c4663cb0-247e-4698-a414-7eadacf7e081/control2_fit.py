"""Fit restored files to their natural reference's length: trim any excess, zero-pad any
shortfall, and report both.
usage: control2_fit.py SCRATCH SOURCE_VARIANT"""
import sys
from pathlib import Path

import torch
import torchaudio

S, V = sys.argv[1], sys.argv[2]
for corpus in ["libritts", "vctk"]:
    src, out = Path(f"{S}/control2/{corpus}/{V}"), Path(f"{S}/control2/{corpus}/{V}_fit")
    out.mkdir(exist_ok=True)
    diff = []
    for f in sorted(src.glob("*.wav")):
        w, sr = torchaudio.load(str(f))
        n = torchaudio.info(f"{S}/control1/{corpus}/sr24000/{f.name}").num_frames
        assert sr == 24000, (f.name, sr)
        diff.append(w.size(1) - n)
        w = torch.nn.functional.pad(w, (0, max(0, n - w.size(1))))[:, :n]
        torchaudio.save(str(out / f.name), w, sr)
    print(corpus, f"length minus reference, samples: min {min(diff)} max {max(diff)}", flush=True)
