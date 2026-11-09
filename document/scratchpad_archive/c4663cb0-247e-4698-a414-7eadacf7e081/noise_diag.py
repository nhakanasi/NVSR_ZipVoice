"""Does the branch remove its state's noise? Teacher-forced, y = t*x1 + (1-t)*eps.

err = x1_hat - x1 on generated frames x fully-missing bins.
  beta   = <err, eps> / <eps, eps>: the noise left in the endpoint.
           Ignoring the noise entirely gives beta = (1-t); perfect removal 0.
  kept   = beta / (1-t): fraction of the state's noise passed to the endpoint.
  mse    = mean err^2, split into the noise-aligned part (beta^2 var eps)
           and the rest. fm_loss in training is this mse, averaged over t.
The backbone (clean prompt) on the same bins is the control.
usage: noise_diag.py N_ITEMS CKPT [CKPT ...]"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from zipvoice.models.zipvoice_dual import ZipVoiceDual
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm

EV = "C:/zipvoice_data/data/bwe_eval"
DL = "C:/zipvoice_data/download/zipvoice"
dev = torch.device("cuda")
N = int(sys.argv[1])
TS = [0.1, 0.3, 0.5, 0.7, 0.85, 0.882, 0.95]  # 0.882: last call of the shifted 16-step grid
tok = EmiliaTokenizer(token_file=f"{DL}/tokens.txt")
fb = VocosFbank()


def feat(path):
    w, sr = load_prompt_wav(path, sampling_rate=24000)
    w, _ = rms_norm(w, 0.1)
    return fb.extract(w, sampling_rate=24000).to(dev) * 0.1, sr


rows = [l.rstrip("\n").split("\t") for l in open(f"{EV}/test_sr8000.tsv", encoding="utf-8")][:N]
items = []
for uid, ptext, ppath, text in rows:
    pf, psr = feat(ppath)
    cf, _ = feat(ppath.replace("sr8000", "clean"))
    rf, _ = feat(f"{EV}/references/{uid}.wav")
    items.append((pf, psr, cf, rf, ptext, text))


def stats(err, eps, t):
    beta = (err * eps).sum() / (eps * eps).sum()
    mse = err.pow(2).mean()
    aligned = beta ** 2 * eps.pow(2).mean()
    return [beta.item() / (1 - t), mse.item(), aligned.item()]


first = True
for ckpt in sys.argv[2:]:
    cfg = json.load(open(ckpt.rsplit("/", 1)[0] + "/model.json"))
    model = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), feat_scale=0.1,
                         sampling_rate=24000, vocab_size=tok.vocab_size, pad_id=tok.pad_id)
    load_checkpoint(ckpt, model=model, strict=True)
    model = model.to(dev).eval()
    R, B = [], []
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        for pf, psr, cf, rf, ptext, text in items:
            P, Tt = pf.size(0), rf.size(0)
            ref = torch.cat([pf, rf])[None]
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
            sel = torch.zeros(1, L, 100, dtype=torch.bool, device=dev)
            sel[:, P:, mask[0].flatten() > 0.99] = True
            eps = torch.randn(ref.shape, device=dev, generator=g)
            r, b = [], []
            for t in TS:
                tt = torch.tensor(t, device=dev)
                y = ref * t + eps * (1 - t)
                h = model._memory(t=tt, y=y, text_condition=tc, speech_condition=sc,
                                  padding_mask=pad)
                x1 = model.branch(t=tt, xt=y, text_condition=tc, speech_condition=sc,
                                  mask=mask, cutoff_hz=cutoff, prompt_mask=pm, memory=h,
                                  padding_mask=pad, condition=cond)
                r.append(stats((x1 - ref)[sel], eps[sel], t))
                if first and cf.size(0) == P:
                    refc = torch.cat([cf, rf])[None]
                    yc = refc * t + eps * (1 - t)
                    scc = F.pad(cf[None], (0, 0, 0, L - P))
                    v, _ = model._backbone_step(t=tt, x=yc, text_condition=tc,
                                                speech_condition=scc, padding_mask=pad,
                                                guidance_scale=0.0)
                    b.append(stats((yc + (1 - t) * v - refc)[sel], eps[sel], t))
            R.append(r)
            if b:
                B.append(b)
    R = np.mean(R, 0)
    print(f"\n{ckpt}  ({cfg.get('dual', {}).get('branch_output_param', 'endpoint')})")
    print("     t   kept   mse     noise-aligned-mse")
    for i, t in enumerate(TS):
        print(f"  {t:4.2f}  {R[i][0]:6.3f}  {R[i][1]:.4f}  {R[i][2]:.4f}")
    if first and B:
        B = np.mean(B, 0)
        print("backbone (clean prompt), same bins")
        for i, t in enumerate(TS):
            print(f"  {t:4.2f}  {B[i][0]:6.3f}  {B[i][1]:.4f}  {B[i][2]:.4f}")
    first = False
    del model
    torch.cuda.empty_cache()
