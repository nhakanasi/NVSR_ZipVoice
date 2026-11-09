"""Summarise inv_check.py output against the pre-registered pass rule."""
import json
import sys

import numpy as np

for path in sys.argv[1:]:
    R = [json.loads(l) for l in open(path, encoding="utf-8")]
    f16 = np.array([r["nmse_fp16"] for r in R])
    f32 = np.array([r["nmse_fp32"] for r in R])
    anc = np.array([r["anchor_nmse"] for r in R])
    med, p90, amed = np.median(f16), np.percentile(f16, 90), np.median(anc)
    ok = med <= 0.05 and p90 <= 0.15 and med <= 0.2 * amed
    print(f"== {path.rsplit('/', 1)[-1]}  n={len(R)}")
    print(f"replay fp16 NMSE median {med:.3f} p90 {p90:.3f} | fp32 median {np.median(f32):.3f} "
          f"p90 {np.percentile(f32, 90):.3f} | anchor median {amed:.3f}  -> {'PASS' if ok else 'FAIL'}")
    print(f"items fp16 <= 0.05: {(f16 <= 0.05).mean():.0%}; |fp16 - fp32| median "
          f"{np.median(np.abs(f16 - f32)):.3f}")
    res = np.array([r["residual"] for r in R])
    it = np.array([r["iters"] for r in R])
    print("residual median per step:", " ".join(f"{x:.0e}" for x in np.median(res, 0)))
    print("share of steps not converged (hit max iters):", f"{(it >= 39).mean():.0%}")
    for k in ("z_std", "z_kurt", "z_ac_time", "z_ac_freq"):
        print(f"{k} median {np.median([r[k] for r in R]):.3f}", end="; ")
    print("(Gaussian: 1, 0, 0, 0)")
    div = np.median(np.array([r["guide_div"] for r in R]), 0)
    print("guided vs unguided low-band divergence per node (x endpoint std):",
          " ".join(f"{x:.2f}" for x in div))
    fib = [r for r in R if "fiber_nmse_fp16" in r]
    if fib:
        fn = np.array([r["fiber_nmse_fp16"] for r in fib])
        fz = np.array([r["fiber_zdist"] for r in fib])
        print(f"fiber (n={len(fib)}): per-item spread of fp16 NMSE over 3 eps, median "
              f"{np.median(fn.max(1) - fn.min(1)):.3f}; z* distance between eps median "
              f"{np.median(fz):.3f}")
