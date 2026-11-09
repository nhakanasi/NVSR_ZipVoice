"""AudioSR pads its output past the input length. Cut each restored file to its natural
reference's length, report how much was cut and how loud the cut tail was.
usage: control2_trim.py SCRATCH"""
import sys
from pathlib import Path

import numpy as np
import torchaudio

S = sys.argv[1]
for corpus in ["libritts", "vctk"]:
    src, out = Path(f"{S}/control2/{corpus}/audiosr_sr8000"), Path(f"{S}/control2/{corpus}/audiosr_sr8000_trim")
    out.mkdir(exist_ok=True)
    extra, tail_db = [], []
    for f in sorted(src.glob("*.wav")):
        w, sr = torchaudio.load(str(f))
        n = torchaudio.info(f"{S}/control1/{corpus}/sr24000/{f.name}").num_frames
        assert sr == 24000 and w.size(1) >= n, (f.name, sr, w.size(1), n)
        extra.append((w.size(1) - n) / sr)
        if w.size(1) > n:
            tail_db.append(10 * np.log10(w[:, n:].pow(2).mean().item() / w[:, :n].pow(2).mean().item() + 1e-12))
        torchaudio.save(str(out / f.name), w[:, :n], sr)
    print(corpus, f"extra s median {np.median(extra):.2f} max {max(extra):.2f}; "
          f"tail level dB re speech median {np.median(tail_db):.1f} max {max(tail_db):.1f}", flush=True)
