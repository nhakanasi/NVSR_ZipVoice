"""Can the frozen backbone's guided 16-step Euler sampler be inverted onto the
ground-truth low band?

For each eval item the endpoint is mel(lowpass(reference, prompt cutoff)) for the
target frames and the band-limited prompt mel for the prompt frames: what the
backbone produces under a band-limited prompt, with the ground-truth low band.
Each backward step solves x_k = x_{k+1} - dt_k * v(x_k, t_k) by damped fixed-point
iteration on the exact grid and guidance the eval sampler uses. The last step
starts from t_k*x1 + (1-t_k)*eps.

Metric: low-band NMSE on target frames, mse(replay, endpoint) / var(endpoint).
Pre-registered pass, per rate, on the fp16 replay (training precision):
median <= 0.05, p90 <= 0.15, median <= 0.2 * unpaired anchor median.

Also reported: per-step residual, fp32 replay, z* typicality, unpaired anchor
(forward from random noise), fiber test (3 eps on the first N_FIBER items), and
guided-vs-unguided low-band divergence per node from the same noise.
usage: inv_check.py CKPT SR N_ITEMS OUT_JSONL"""
import json
import sys
import time

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.zipvoice_dual import ZipVoiceDual
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm

EV = "C:/zipvoice_data/data/bwe_eval"
DL = "C:/zipvoice_data/download/zipvoice"
NUM_STEP, T_SHIFT, GUIDE = 16, 0.5, 1.0
MAX_IT, TOL = 40, 1e-4
N_FIBER = 10
dev = torch.device("cuda")
ckpt, sr_tag, n_items, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(ckpt.rsplit("/", 1)[0] + "/model.json"))
tok = EmiliaTokenizer(token_file=f"{DL}/tokens.txt")
model = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), feat_scale=0.1,
                     sampling_rate=24000, vocab_size=tok.vocab_size, pad_id=tok.pad_id)
load_checkpoint(ckpt, model=model, strict=True)
model = model.to(dev).eval()
fb = VocosFbank()
ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=NUM_STEP, t_shift=T_SHIFT, device=dev)


def wav(path):
    w, sr = load_prompt_wav(path, sampling_rate=24000)
    w, _ = rms_norm(w, 0.1)
    return w, sr


def mel(w):
    return fb.extract(w, sampling_rate=24000).to(dev) * 0.1


def rms(a):
    return a.pow(2).mean().sqrt().item()


rows = [l.rstrip("\n").split("\t") for l in open(f"{EV}/test_{sr_tag}.tsv", encoding="utf-8")]
out = open(out_path, "w", encoding="utf-8")
torch.manual_seed(0)

with torch.no_grad():
    for i, (uid, ptext, ppath, text) in enumerate(rows[:n_items]):
        t0 = time.time()
        pw, psr = wav(ppath)
        pf = mel(pw)
        rw, _ = wav(f"{EV}/references/{uid}.wav")
        rw_lp = AF.resample(AF.resample(rw, 24000, psr), psr, 24000)
        rf = mel(rw_lp)
        P, Tt = pf.size(0), rf.size(0)
        x1 = torch.cat([pf, rf])[None]
        tc, pad = model.forward_text_inference_gt_duration(
            tok.texts_to_token_ids([text]), torch.tensor([Tt], device=dev),
            tok.texts_to_token_ids([ptext]), torch.tensor([P], device=dev))
        L = tc.size(1)
        sc = F.pad(pf[None], (0, 0, 0, L - P))
        mask = model._band_mask(torch.tensor([psr / 2], device=dev))
        lo = mask[0].flatten() < 0.01
        end_lo = x1[0, P:][:, lo]
        var_lo = end_lo.var().item()

        def v(x, t, g=GUIDE):
            return model._backbone_step(t=t, x=x, text_condition=tc, speech_condition=sc,
                                        padding_mask=pad, guidance_scale=g)[0].float()

        def forward(z, g=GUIDE, fp16=False, keep=False):
            x, states = z, []
            for k in range(NUM_STEP):
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    vk = v(x, ts[k], g)
                x = x + (ts[k + 1] - ts[k]) * vk
                if keep:
                    states.append(x[0, P:][:, lo])
            return (x, states) if keep else x

        def nmse(x):
            return ((x[0, P:][:, lo] - end_lo) ** 2).mean().item() / var_lo

        def invert(eps):
            x_next, res, its = x1, [], []
            for k in reversed(range(NUM_STEP)):
                dt = ts[k + 1] - ts[k]
                x = ts[k] * x1 + (1 - ts[k]) * eps if k == NUM_STEP - 1 \
                    else x_next - dt * v(x_next, ts[k])
                alpha, prev = 1.0 if k < NUM_STEP - 1 else 0.5, float("inf")
                for it in range(MAX_IT):
                    r = x + dt * v(x, ts[k]) - x_next
                    rr = rms(r) / rms(x_next)
                    if rr < TOL:
                        break
                    if rr > prev:
                        alpha *= 0.5
                    prev = rr
                    x = x - alpha * r
                res.append(rr)
                its.append(it)
                x_next = x
            return x_next, res[::-1], its[::-1]

        z, res, its = invert(torch.randn_like(x1))
        zt = z[0, P:]
        zc = zt - zt.mean()
        rec = {
            "uid": uid, "frames": Tt,
            "nmse_fp32": nmse(forward(z)),
            "nmse_fp16": nmse(forward(z, fp16=True)),
            "residual": res, "iters": its,
            "z_mean": zt.mean().item(), "z_std": zt.std().item(),
            "z_kurt": (zc.pow(4).mean() / zc.pow(2).mean() ** 2).item() - 3,
            "z_ac_time": F.cosine_similarity(zc[1:].flatten(), zc[:-1].flatten(), 0).item(),
            "z_ac_freq": F.cosine_similarity(zc[:, 1:].flatten(), zc[:, :-1].flatten(), 0).item(),
        }
        eps = torch.randn_like(x1)
        xg, sg = forward(eps, keep=True)
        _, su = forward(eps, g=0.0, keep=True)
        rec["anchor_nmse"] = nmse(xg)
        std_lo = var_lo ** 0.5
        rec["guide_div"] = [rms(a - b) / std_lo for a, b in zip(sg, su)]
        if i < N_FIBER:
            zs = [z] + [invert(torch.randn_like(x1))[0] for _ in range(2)]
            rec["fiber_nmse_fp16"] = [nmse(forward(zz, fp16=True)) for zz in zs]
            rec["fiber_zdist"] = [rms(zs[a] - zs[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
        rec["sec"] = time.time() - t0
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{i} {uid} nmse16 {rec['nmse_fp16']:.4f} anchor {rec['anchor_nmse']:.3f} "
              f"res_last {res[-1]:.1e} it_last {its[-1]} {rec['sec']:.0f}s", flush=True)
