"""The cascade on a real system: stock ZipVoice given NVSR-restored 8 kHz prompts, paired
against stock at the raw 8 kHz and 24 kHz prompts and against natural speech, per verifier.
Also the energy the cascade's output carries in the bands the prompt lacked.
usage: cascade_compare.py SCRATCH"""
import sys

import numpy as np
import torch
import torchaudio

S = sys.argv[1]
RS = "C:/zipvoice_data/exp/results_rs"
SYSTEMS = {"nat24": f"{S}/control1/{{c}}/sr24000", "nat8": f"{S}/control1/{{c}}/sr8000",
           "stock24": f"{RS}/{{c}}/stock_sr24000", "stock8": f"{RS}/{{c}}/stock_sr8000",
           "cascade8": f"{RS}/{{c}}/cascade_nvsr_sr8000"}
PAIRS = [("cascade8", "stock8"), ("stock24", "cascade8"), ("nat8", "cascade8")]
rng = np.random.default_rng(0)


def load(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")][1:]
    return {u: float(v) for u, v in rows}


def band_db(path, lo, hi):
    w, sr = torchaudio.load(path)
    p = torch.stft(w[0], 2048, 512, window=torch.hann_window(2048), return_complex=True).abs().pow(2).sum(1)
    f = torch.fft.rfftfreq(2048, 1 / sr)
    return 10 * np.log10((p[(f > lo) & (f <= hi)].sum() / p.sum()).item() + 1e-12)


for corpus in ["libritts", "vctk"]:
    for score in ["sim_scores.tsv", "campp_scores.tsv"]:
        d = {k: load(v.format(c=corpus) + "/" + score) for k, v in SYSTEMS.items()}
        uids = sorted(d["nat24"])
        assert all(sorted(x) == uids for x in d.values())
        a = {k: np.array([x[u] for u in uids]) for k, x in d.items()}
        idx = [rng.integers(0, len(uids), len(uids)) for _ in range(10000)]
        print(f"== {corpus} {score}  means " + " ".join(f"{k} {v.mean():.4f}" for k, v in a.items()))
        for x, y in PAIRS:
            dd = a[x] - a[y]
            b = np.array([dd[i].mean() for i in idx])
            print(f"  {x} - {y:10s} {dd.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, "
                  f"{np.percentile(b, 97.5):+.3f}]  {x} lower in {np.mean(dd < 0):.0%}")
    for lo, hi in [(4200, 8000), (8000, 12000)]:
        e = {k: np.median([band_db(v.format(c=corpus) + f"/{u}.wav", lo, hi) for u in uids])
             for k, v in SYSTEMS.items()}
        print(f"  energy {lo}-{hi} Hz dB re total (median): " + " ".join(f"{k} {v:.1f}" for k, v in e.items()))
