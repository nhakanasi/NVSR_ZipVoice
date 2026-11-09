"""Control 2 paired summary: AudioSR-restored 8 kHz natural references against the natural
references at 24 kHz and at 8 kHz, per verifier, with a paired bootstrap; plus a sanity
check that the restored files are the same length and carry a high band.
usage: control2_summary.py SCRATCH VARIANT [VARIANT ...]   (dirs under control2/CORPUS/)"""
import sys

import numpy as np
import torch
import torchaudio

S = sys.argv[1]
rng = np.random.default_rng(0)


def load(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")][1:]
    return {u: float(v) for u, v in rows}


def band_db(path, lo, hi):
    w, sr = torchaudio.load(path)
    p = torch.stft(w[0], 2048, 512, window=torch.hann_window(2048), return_complex=True).abs().pow(2).sum(1)
    f = torch.fft.rfftfreq(2048, 1 / sr)
    return 10 * np.log10((p[(f > lo) & (f <= hi)].sum() / p.sum()).item() + 1e-12), w.size(1) / sr


for corpus in ["libritts", "vctk"]:
    dirs = {"nat24": f"{S}/control1/{corpus}/sr24000", "nat8": f"{S}/control1/{corpus}/sr8000",
            **{v: f"{S}/control2/{corpus}/{v}" for v in sys.argv[2:]}}
    for score in ["sim_scores.tsv", "campp_scores.tsv"]:
        d = {k: load(f"{v}/{score}") for k, v in dirs.items()}
        uids = sorted(d["nat24"])
        a = {k: np.array([x[u] for u in uids]) for k, x in d.items()}
        idx = [rng.integers(0, len(uids), len(uids)) for _ in range(10000)]
        print(f"== {corpus} {score} means " + " ".join(f"{k} {v.mean():.4f}" for k, v in a.items()))
        for v in sys.argv[2:]:
            for name, dif in {f"{v} - nat8": a[v] - a["nat8"], f"nat24 - {v}": a["nat24"] - a[v]}.items():
                b = np.array([dif[i].mean() for i in idx])
                print(f"  {name:40s} {dif.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
            print(f"  {v} lower than nat8 in {np.mean(a[v] < a['nat8']):.0%} of items")
    for lo, hi in [(4200, 8000), (8000, 12000)]:
        r = {k: [band_db(f"{v}/{u}.wav", lo, hi) for u in uids] for k, v in dirs.items()}
        print(f"  energy {lo}-{hi} Hz dB re total (median): "
              + " ".join(f"{k} {np.median([x[0] for x in v]):.1f}" for k, v in r.items()))
    dur = {k: np.array([x[1] for x in v]) for k, v in r.items()}
    print(f"  max |duration variant - nat24| {np.abs(dur[sys.argv[2]] - dur['nat24']).max():.3f} s")
