"""Does what the branch is given carry the target's high-band shape?

Linear (ridge) probes from each input stream, per target frame, to the
reference high band (bins with mask > 0.99), fitted on some utterances and
scored (R^2) on held-out ones. Two targets:
  level  per-frame mean over the high bins (the envelope)
  shape  per-frame deviation of each high bin from that mean (the spectrum)
Both are centred on the training frames' mean, so R^2 = 0 is "predict the
average frame" and a fixed per-bin shape scores 0 on `shape`.

Streams (all at target frames; state layout [prompt | target], as in eval):
  text        text_condition, the uniformly upsampled token embedding
  low_true    reference low band, +-2 frames (upper bound for the low band)
  mem_roll    hidden on the branch's own solve state at t (true low band,
              branch-made high band): what the branch reads at inference
  mem_nolow   hidden on y = pure noise everywhere (text + prompt only)
  mem_full    hidden on the full teacher state at t (high band = t*ref+(1-t)eps):
              what teacher-forced training shows the branch
  cond        encode_condition output (freq/speaker/rate streams)
  branch      the branch's endpoint high band at that step; solve_out = final
  ... raw     the same, scored directly (no fitted readout)
usage: cond_probe.py CKPT"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from zipvoice.models.zipvoice_dual import ZipVoiceDual, endpoint_to_velocity
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm

EV = "C:/zipvoice_data/data/bwe_eval"
DL = "C:/zipvoice_data/download/zipvoice"
TS = [0.3, 0.6, 0.85]
N_TRAIN = 50
dev = torch.device("cuda")
ckpt = sys.argv[1]
cfg = json.load(open(ckpt.rsplit("/", 1)[0] + "/model.json"))
tok = EmiliaTokenizer(token_file=f"{DL}/tokens.txt")
model = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), feat_scale=0.1,
                     sampling_rate=24000, vocab_size=tok.vocab_size, pad_id=tok.pad_id)
load_checkpoint(ckpt, model=model, strict=True)
model = model.to(dev).eval()
fb = VocosFbank()


def feat(path):
    w, sr = load_prompt_wav(path, sampling_rate=24000)
    w, _ = rms_norm(w, 0.1)
    return fb.extract(w, sampling_rate=24000).to(dev) * 0.1, sr


rows = [l.rstrip("\n").split("\t") for l in open(f"{EV}/test_sr8000.tsv", encoding="utf-8")]
X = {}  # stream -> list of (Tt, D) per item
Y = []  # list of (Tt, H) reference high band per item
g = torch.Generator(device=dev).manual_seed(0)


def add(name, v):
    X.setdefault(name, []).append(v.float().cpu())


with torch.no_grad():
    for uid, ptext, ppath, text in rows:
        pf, psr = feat(ppath)
        rf, _ = feat(f"{EV}/references/{uid}.wav")
        P, Tt = pf.size(0), rf.size(0)
        ref = torch.cat([pf, rf])[None]
        tc, pad = model.forward_text_inference_gt_duration(
            tok.texts_to_token_ids([text]), torch.tensor([Tt], device=dev),
            tok.texts_to_token_ids([ptext]), torch.tensor([P], device=dev))
        L = tc.size(1)
        sc = F.pad(pf[None], (0, 0, 0, L - P))
        cutoff = torch.tensor([psr / 2], device=dev)
        mask = model._band_mask(cutoff)
        hi = mask[0].flatten() > 0.99
        lo = mask[0].flatten() < 0.01
        pm = torch.zeros(1, L, dtype=torch.bool, device=dev)
        pm[:, :P] = True
        cond = model.branch.encode_condition(speech_condition=sc, mask=mask,
                                             cutoff_hz=cutoff, prompt_mask=pm)
        tgt = slice(P, L)
        Y.append(rf[:, hi].float().cpu())
        add("text", tc[0, tgt])
        low = F.pad(rf[:, lo].T[None], (2, 2), mode="replicate")[0].T  # (Tt+4, nlo)
        add("low_true", torch.cat([low[k:k + Tt] for k in range(5)], 1))
        add("cond", cond[0, tgt])
        eps = torch.randn(ref.shape, device=dev, generator=g)
        for t in TS:
            tt = torch.tensor(t, device=dev)
            full = ref * t + eps * (1 - t)
            for name, y in (("nolow", eps), ("full", full)):
                h = model._memory(t=tt, y=y, text_condition=tc,
                                  speech_condition=sc, padding_mask=pad)
                add(f"mem_{name}@{t}", h[0, tgt])
        # Aligned solve: the low band follows the teacher path (true low band),
        # the high band is the branch's own trajectory from pure noise, exactly
        # as it evolves at inference. Same grid as eval.
        ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=16, t_shift=0.5, device=dev)
        want = {int((ts[:-1] - t).abs().argmin()): t for t in TS}
        y = eps
        for k in range(16):
            tk, dt = ts[k], ts[k + 1] - ts[k]
            y = model._project(ref * tk + eps * (1 - tk), y, mask)
            h = model._memory(t=tk, y=y, text_condition=tc, speech_condition=sc,
                              padding_mask=pad)
            b = model.branch(t=tk, xt=y, text_condition=tc, speech_condition=sc,
                             mask=mask, cutoff_hz=cutoff, prompt_mask=pm, memory=h,
                             padding_mask=pad, condition=cond)
            if k in want:
                add(f"mem_roll@{want[k]}", h[0, tgt])
                add(f"branch@{want[k]}", b[0, tgt][:, hi])
            y = y + endpoint_to_velocity(b, y, tk) * dt
        add("solve_out", y[0, tgt][:, hi])
        print(".", end="", flush=True)
print()


def split(y):
    lvl = y.mean(1, keepdim=True)
    return lvl, y - lvl


def r2(pred, true, mu):
    return 1 - ((pred - true) ** 2).sum().item() / ((true - mu) ** 2).sum().item()


def ridge(xtr, ytr, xte, lam):
    m, s = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = (xtr - m) / s, (xte - m) / s
    a = torch.cat([a, torch.ones(len(a), 1, dtype=a.dtype)], 1)
    b = torch.cat([b, torch.ones(len(b), 1, dtype=b.dtype)], 1)
    w = torch.linalg.solve(a.T @ a + lam * torch.eye(a.size(1), dtype=a.dtype), a.T @ ytr)
    return b @ w


for t in TS:
    X[f'avail@{t}'] = [torch.cat(z, 1) for z in zip(X['cond'], X[f'mem_roll@{t}'])]
X['low+cond'] = [torch.cat(z, 1) for z in zip(X['cond'], X['low_true'])]
ytr, yte = torch.cat(Y[:N_TRAIN]).double(), torch.cat(Y[N_TRAIN:]).double()
targets = {"level": (split(ytr)[0], split(yte)[0]), "shape": (split(ytr)[1], split(yte)[1])}
# lambda chosen on a split inside the training items
n_in = sum(len(y) for y in Y[:35])
print(f"items train {N_TRAIN} test {len(Y) - N_TRAIN}; frames {len(ytr)}/{len(yte)}; "
      f"high bins {Y[0].size(1)}")
print(f"{'stream':22s} {'dim':>5s} {'level R2':>9s} {'shape R2':>9s}")
for name, xs in X.items():
    xtr, xte = torch.cat(xs[:N_TRAIN]).double(), torch.cat(xs[N_TRAIN:]).double()
    res = []
    for key, (a, b) in targets.items():
        best = max((r2(ridge(xtr[:n_in], a[:n_in], xtr[n_in:], lam), a[n_in:],
                       a[:n_in].mean(0)), lam) for lam in (1e0, 1e1, 1e2, 1e3, 1e4))[1]
        res.append(r2(ridge(xtr, a, xte, best), b, a.mean(0)))
    print(f"{name:22s} {xtr.size(1):5d} {res[0]:9.3f} {res[1]:9.3f}")
    if name.startswith("branch@") or name == "solve_out":
        bl, bs = split(xte)
        print(f"{name + ' raw':22s} {'':5s} "
              f"{r2(bl, targets['level'][1], targets['level'][0].mean(0)):9.3f} "
              f"{r2(bs, targets['shape'][1], targets['shape'][0].mean(0)):9.3f}")
