"""Control 1 decomposition with per-utterance pairing, for each verifier and corpus.

Inputs per corpus: natural references at 24 kHz and cut to 8 kHz (control1/), and stock
ZipVoice resynthesised from 24 kHz and 8 kHz prompts (results_rs/). Every row is a
difference of means over the same 76 utterances, so the paired bootstrap resamples
utterances once and applies the same draw to all four systems.

Also checks the additivity premise: is stock's output at an 8 kHz prompt band-limited the
way the prompt is? Reported as the share of energy above 4.2 kHz, in dB.
usage: decompose.py SCRATCH"""
import sys

import numpy as np
import torch
import torchaudio

S = sys.argv[1]
RS = "C:/zipvoice_data/exp/results_rs"
SYSTEMS = {"nat24": f"{S}/control1/{{c}}/sr24000", "nat8": f"{S}/control1/{{c}}/sr8000",
           "stock24": f"{RS}/{{c}}/stock_sr24000", "stock8": f"{RS}/{{c}}/stock_sr8000"}
rng = np.random.default_rng(0)


def load(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")][1:]
    return {u: float(v) for u, v in rows}


def ci(stat, idx):
    b = np.array([stat(i) for i in idx])
    return f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"


def high_share_db(path, cut=4200):
    w, sr = torchaudio.load(path)
    spec = torch.stft(w[0], 2048, 512, window=torch.hann_window(2048), return_complex=True)
    p = spec.abs().pow(2).sum(1)
    f = torch.fft.rfftfreq(2048, 1 / sr)
    return 10 * np.log10((p[f > cut].sum() / p.sum()).item() + 1e-12)


for corpus in ["libritts", "vctk"]:
    for score in ["sim_scores.tsv", "campp_scores.tsv"]:
        d = {k: load(v.format(c=corpus) + "/" + score) for k, v in SYSTEMS.items()}
        uids = sorted(d["nat24"])
        assert all(sorted(x) == uids for x in d.values())
        a = {k: np.array([x[u] for u in uids]) for k, x in d.items()}
        idx = [rng.integers(0, len(uids), len(uids)) for _ in range(10000)]
        rows = {
            "bandwidth effect (nat24 - nat8)": lambda i: a["nat24"][i].mean() - a["nat8"][i].mean(),
            "stock shortfall, full band (nat24 - stock24)": lambda i: a["nat24"][i].mean() - a["stock24"][i].mean(),
            "stock shortfall, both 8k (nat8 - stock8)": lambda i: a["nat8"][i].mean() - a["stock8"][i].mean(),
            "identity lost ((nat8-stock8) - (nat24-stock24))": lambda i: (a["nat8"][i] - a["stock8"][i] - a["nat24"][i] + a["stock24"][i]).mean(),
            "stock 8k drop (stock24 - stock8)": lambda i: a["stock24"][i].mean() - a["stock8"][i].mean(),
            "registered share (nat24-nat8)/(nat24-stock8)": lambda i: (a["nat24"][i].mean() - a["nat8"][i].mean()) / (a["nat24"][i].mean() - a["stock8"][i].mean()),
            "corrected share (nat24-nat8)/(stock24-stock8)": lambda i: (a["nat24"][i].mean() - a["nat8"][i].mean()) / (a["stock24"][i].mean() - a["stock8"][i].mean()),
        }
        full = np.arange(len(uids))
        print(f"== {corpus} {score}  means " + " ".join(f"{k} {v.mean():.4f}" for k, v in a.items()))
        for name, f in rows.items():
            print(f"  {name:50s} {f(full):.3f} {ci(f, idx)}")
        reg = np.array([rows["registered share (nat24-nat8)/(nat24-stock8)"](i) for i in idx])
        print(f"  P(registered share >= 2/3) {np.mean(reg >= 2 / 3):.2f}")
    hs = {k: np.array([high_share_db(v.format(c=corpus) + f"/{u}.wav") for u in uids])
          for k, v in SYSTEMS.items()}
    print(f"== {corpus} energy above 4.2 kHz, dB re total (median): "
          + " ".join(f"{k} {np.median(v):.1f}" for k, v in hs.items()))
