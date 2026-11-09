"""Mel of the eval-matched 16-step solve for a few sr8000 items, several
checkpoints side by side, plus one frame's spectrum above the cutoff.
usage: plot_solve_mel.py OUT.png ITEM_INDEX LABEL=CKPT [LABEL=CKPT ...]"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from zipvoice.models.zipvoice_dual import ZipVoiceDual
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.zipvoice_dual import endpoint_to_velocity

EV = "C:/zipvoice_data/data/bwe_eval"
DL = "C:/zipvoice_data/download/zipvoice"
dev = torch.device("cuda")
out_png, idx = sys.argv[1], int(sys.argv[2])
systems = [a.split("=", 1) for a in sys.argv[3:]]
tok = EmiliaTokenizer(token_file=f"{DL}/tokens.txt")
fb = VocosFbank()


def feat(path):
    w, sr = load_prompt_wav(path, sampling_rate=24000)
    w, _ = rms_norm(w, 0.1)
    return fb.extract(w, sampling_rate=24000).to(dev) * 0.1, sr


rows = [l.rstrip("\n").split("\t") for l in open(f"{EV}/test_sr8000.tsv", encoding="utf-8")]
uid, ptext, ppath, text = rows[idx]
pf, psr = feat(ppath)
rf, _ = feat(f"{EV}/references/{uid}.wav")
P, Tt = pf.size(0), rf.size(0)
mels = [("reference", rf.cpu().numpy() / 0.1)]

for label, ckpt in systems:
    cfg = json.load(open(ckpt.rsplit("/", 1)[0] + "/model.json"))
    model = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), feat_scale=0.1,
                         sampling_rate=24000, vocab_size=tok.vocab_size,
                         pad_id=tok.pad_id)
    load_checkpoint(ckpt, model=model, strict=True)
    model = model.to(dev).eval()
    with torch.no_grad():
        tc, pad = model.forward_text_inference_gt_duration(
            tok.texts_to_token_ids([text]), torch.tensor([Tt], device=dev),
            tok.texts_to_token_ids([ptext]), torch.tensor([P], device=dev))
        L = tc.size(1)
        sc = F.pad(pf[None], (0, 0, 0, L - P))
        cutoff = torch.tensor([psr / 2], device=dev)
        mask = model._band_mask(cutoff)
        pm = torch.zeros(1, L, dtype=torch.bool, device=dev)
        pm[:, :P] = True
        cond = model.branch.encode_condition(speech_condition=sc, mask=mask,
                                             cutoff_hz=cutoff, prompt_mask=pm)
        ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=16, t_shift=0.5, device=dev)
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(1, L, 100, device=dev, generator=g)
        y = x
        for k in range(16):
            tk, dt = ts[k], ts[k + 1] - ts[k]
            v, _ = model._backbone_step(t=tk, x=x, text_condition=tc, speech_condition=sc,
                                        padding_mask=pad, guidance_scale=1.0)
            h = model._memory(t=tk, y=y, text_condition=tc, speech_condition=sc,
                              padding_mask=pad)
            b1 = model.branch(t=tk, xt=y, text_condition=tc, speech_condition=sc,
                              mask=mask, cutoff_hz=cutoff, prompt_mask=pm, memory=h,
                              padding_mask=pad, condition=cond)
            y = model._project(x + v * dt, y + endpoint_to_velocity(b1, y, tk) * dt, mask)
            x = x + v * dt
        o = model._project(x, y, mask)[0, P:]
    mels.append((label, o.cpu().numpy() / 0.1))
    del model
    torch.cuda.empty_cache()

# cutoff bin: first bin whose mask exceeds 0.5
cut_bin = int((mask[0].flatten() > 0.5).nonzero()[0])
n = len(mels)
fig, axes = plt.subplots(n + 1, 1, figsize=(12, 2.6 * (n + 1)))
vmin, vmax = mels[0][1].min(), mels[0][1].max()
# a voiced frame: highest low-band energy in the reference
fr = int(mels[0][1][:, :cut_bin].mean(1).argmax())
for ax, (label, m) in zip(axes, mels):
    T = min(m.shape[0], mels[0][1].shape[0])
    ax.imshow(m[:T].T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma")
    ax.axhline(cut_bin, color="c", lw=0.8, ls="--")
    ax.axvline(fr, color="w", lw=0.6, ls=":")
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("mel bin")
ax = axes[-1]
for label, m in mels:
    ax.plot(np.arange(100), m[min(fr, m.shape[0] - 1)], label=label, lw=1.2)
ax.axvline(cut_bin, color="k", lw=0.8, ls="--")
ax.set_xlabel("mel bin (dashed = 4 kHz cutoff)")
ax.set_ylabel("log-mel")
ax.set_title(f"spectrum of frame {fr} (dotted line above)", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle(f"{uid}  sr8000 prompt, 16-step solve, seed 0", fontsize=11)
fig.tight_layout()
fig.savefig(out_png, dpi=110)
print(out_png, "cut_bin", cut_bin, "frame", fr)
