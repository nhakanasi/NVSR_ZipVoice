"""Mean band levels (dB) over the eval set for any result systems: band_any.py RATE SYS..."""
import sys
from pathlib import Path
import numpy as np
from band_levels import spec_db, bands, R, REF

rate, systems = int(sys.argv[1]), sys.argv[2:]
cut = rate // 2
names = sorted(p.stem for p in (R / f"{systems[0]}_sr{rate}").glob("*.wav"))
print(f"sr{rate} (cutoff {cut} Hz), {len(names)} utts")
for s in ["ref", "prompt"] + systems:
    rows = []
    for n in names:
        p = (REF / f"{n}.wav" if s == "ref" else
             Path(f"C:/zipvoice_data/data/bwe_eval/prompts/sr{rate}/{n}.wav") if s == "prompt" else
             R / f"{s}_sr{rate}" / f"{n}.wav")
        if p.exists():
            rows.append(bands(*spec_db(p), cut))
    if rows:
        print(f"  {s:24s}", "  ".join(f"{b} {np.mean([r[b] for r in rows]):6.1f}" for b in rows[0]), f"(n={len(rows)})")
