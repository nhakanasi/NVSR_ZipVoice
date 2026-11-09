"""Forward pairing check: does swapping the band-limited prompt for the clean one
move the backbone's low band, from the same Gaussian noise?

Guided 16-step Euler, exactly the eval sampler, backbone only. Per item, one
noise eps and four solves:
  bl      band-limited prompt (the trajectory the branch sees at inference)
  clean   clean 24 kHz prompt (its high band would be the training target)
  bl16    band-limited prompt under fp16 autocast (precision sensitivity)
  bl_eps2 band-limited prompt, a second noise (scale: unpaired low bands)
Metric: low-band NMSE on target frames, mse(a, b) / var(b), with the low band
the bins below the band-limited cutoff. Pass marks as for the inversion check:
median <= 0.05, p90 <= 0.15.
usage: swap_check.py CKPT SR N_ITEMS"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.zipvoice_dual import ZipVoiceDual
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm

EV = "C:/zipvoice_data/data/bwe_eval"
DL = "C:/zipvoice_data/download/zipvoice"
dev = torch.device("cuda")
ckpt, sr_tag, n_items = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(ckpt.rsplit("/", 1)[0] + "/model.json"))
tok = EmiliaTokenizer(token_file=f"{DL}/tokens.txt")
model = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), feat_scale=0.1,
                     sampling_rate=24000, vocab_size=tok.vocab_size, pad_id=tok.pad_id)
load_checkpoint(ckpt, model=model, strict=True)
model = model.to(dev).eval()
fb = VocosFbank()
ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=16, t_shift=0.5, device=dev)


def feat(path):
    w, sr = load_prompt_wav(path, sampling_rate=24000)
    w, _ = rms_norm(w, 0.1)
    return fb.extract(w, sampling_rate=24000).to(dev) * 0.1, sr


def load(tag):
    return [l.rstrip("\n").split("\t") for l in open(f"{EV}/test_{tag}.tsv", encoding="utf-8")]


rows, clean_rows = load(sr_tag), {r[0]: r for r in load("sr24000")}
g = torch.Generator(device=dev).manual_seed(0)
R = []
with torch.no_grad():
    for uid, ptext, ppath, text in rows[:n_items]:
        pf, psr = feat(ppath)
        cf, _ = feat(clean_rows[uid][2])
        rf, _ = feat(f"{EV}/references/{uid}.wav")
        P, Tt = pf.size(0), rf.size(0)
        assert cf.size(0) == P, (cf.size(0), P)
        tc, pad = model.forward_text_inference_gt_duration(
            tok.texts_to_token_ids([text]), torch.tensor([Tt], device=dev),
            tok.texts_to_token_ids([ptext]), torch.tensor([P], device=dev))
        L = tc.size(1)
        lo = model._band_mask(torch.tensor([psr / 2], device=dev))[0].flatten() < 0.01

        def solve(prompt, eps, fp16=False):
            sc = F.pad(prompt[None], (0, 0, 0, L - P))
            x, states = eps, []
            for k in range(16):
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    v = model._backbone_step(t=ts[k], x=x, text_condition=tc,
                                             speech_condition=sc, padding_mask=pad,
                                             guidance_scale=1.0)[0].float()
                x = x + (ts[k + 1] - ts[k]) * v
                states.append(x[0, P:][:, lo])
            return states

        eps = torch.randn(1, L, pf.size(1), device=dev, generator=g)
        eps2 = torch.randn(1, L, pf.size(1), device=dev, generator=g)
        bl, clean = solve(pf, eps), solve(cf, eps)
        bl16, bl2 = solve(pf, eps, fp16=True), solve(pf, eps2)
        end = clean[-1]
        var, std = end.var().item(), end.std().item()

        def nmse(a, b=end):
            return ((a - b) ** 2).mean().item() / var

        R.append({
            "swap": nmse(bl[-1]),
            "swap_node": [((a - b) ** 2).mean().sqrt().item() / std for a, b in zip(bl, clean)],
            "fp16": nmse(bl16[-1], bl[-1]),
            "noise2": nmse(bl2[-1], bl[-1]),
            "ref": nmse(bl[-1], rf[:, lo]) * var / rf[:, lo].var().item(),
        })
        print(len(R), f"swap {R[-1]['swap']:.3f} fp16 {R[-1]['fp16']:.3f}", flush=True)
print()


def q(k):
    a = np.array([r[k] for r in R])
    return f"median {np.median(a):.3f} p90 {np.percentile(a, 90):.3f}"


sw = np.array([r["swap"] for r in R])
ok = np.median(sw) <= 0.05 and np.percentile(sw, 90) <= 0.15
print(f"== {sr_tag} n={len(R)}")
print(f"prompt swap, endpoint low-band NMSE: {q('swap')}  -> {'PASS' if ok else 'FAIL'}")
print(f"fp16 vs fp32 (band-limited prompt):  {q('fp16')}")
print(f"two noises (unpaired scale):         {q('noise2')}")
print(f"band-limited output vs reference:    {q('ref')}")
node = np.median(np.array([r["swap_node"] for r in R]), 0)
print("swap divergence per node (x endpoint std):", " ".join(f"{x:.2f}" for x in node))
