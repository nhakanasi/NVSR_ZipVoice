"""Why has the branch's high band no structure? Probes on real eval items.

Layout per item, as training builds it: [8 kHz prompt | full-band reference of
the target text], text = prompt text + target text, duration real.

Detail = deviation of the log-mel from its local mean (time: 11-frame moving
average per bin; freq: 5-bin moving average per frame), measured on target
frames x fully-missing bins (mask > 0.99).

  tf    teacher-forced endpoint at t: state = t*ref + (1-t)*noise.
        ratio = std(pred detail)/std(ref detail), corr = corr(pred, ref detail)
  swap  at t=0.6 the branch state's high band is built from the reference
        rolled in time. corr_ref vs corr_swap says whose detail the output
        follows: its own state's (swap) or the conditioning's (ref).
  solve 16-step eval solve, 4 seeds: detail ratio of the output, per-step
        endpoint ratio, and across-seed std / temporal std per band.
  bb    released backbone alone, clean 24 kHz prompt: the same numbers for a
        model that does generate a full band (the ceiling).
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from zipvoice.models.zipvoice_dual import ZipVoiceDual, endpoint_to_velocity
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm
from zipvoice.models.modules.solver import get_time_steps  # noqa

EV = "C:/zipvoice_data/data/bwe_eval"
DL = "C:/zipvoice_data/download/zipvoice"
N_ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
dev = torch.device("cuda")
torch.manual_seed(0)

ckpt = sys.argv[1]
exp = ckpt.rsplit("/", 1)[0]
cfg = json.load(open(f"{exp}/model.json"))
tok = EmiliaTokenizer(token_file=f"{DL}/tokens.txt")
model = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), feat_scale=0.1,
                     sampling_rate=cfg["feature"]["sampling_rate"],
                     vocab_size=tok.vocab_size, pad_id=tok.pad_id)
load_checkpoint(ckpt, model=model, strict=True)
model = model.to(dev).eval()
fb = VocosFbank()


def feat(path):
    w, sr = load_prompt_wav(path, sampling_rate=24000)
    w, _ = rms_norm(w, 0.1)
    return fb.extract(w, sampling_rate=24000).to(dev) * 0.1, sr


def detail(m):
    # m: (T, F) -> temporal and spectral detail
    t = m.T.unsqueeze(0)
    tm = F.avg_pool1d(F.pad(t, (5, 5), mode="replicate"), 11, 1)[0].T
    fm = F.avg_pool1d(F.pad(m.unsqueeze(0), (2, 2), mode="replicate"), 5, 1)[0]
    return m - tm, m - fm


def cmp(pred, ref, sel):
    out = []
    for a, b in zip(detail(pred), detail(ref)):
        a, b = a[sel], b[sel]
        out += [(a.std() / b.std()).item(),
                torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()]
    return out  # tratio, tcorr, fratio, fcorr


def ratio_only(pred, ref, sel):
    return [(a[sel].std() / b[sel].std()).item()
            for a, b in zip(detail(pred), detail(ref))]


rows = [l.rstrip("\n").split("\t") for l in open(f"{EV}/test_sr8000.tsv", encoding="utf-8")]
TS = [0.2, 0.4, 0.6, 0.8, 0.95]
R = {k: [] for k in ["tf", "swap", "solve", "steps", "div", "bb", "bbtf", "lvl"]}

with torch.no_grad():
    for uid, ptext, ppath, text in rows[:N_ITEMS]:
        pf, psr = feat(ppath)
        cf, _ = feat(ppath.replace("sr8000", "clean"))
        rf, _ = feat(f"{EV}/references/{uid}.wav")
        P, Tt = pf.size(0), rf.size(0)
        ref = torch.cat([pf, rf])[None]  # target frames carry the reference
        tokens = tok.texts_to_token_ids([text])
        ptokens = tok.texts_to_token_ids([ptext])
        plen = torch.tensor([P], device=dev)
        tc, pad = model.forward_text_inference_gt_duration(
            tokens, torch.tensor([Tt], device=dev), ptokens, plen)
        L = tc.size(1)
        sc = F.pad(pf[None], (0, 0, 0, L - P))
        cutoff = torch.tensor([psr / 2], device=dev)
        mask = model._band_mask(cutoff)
        pm = torch.zeros(1, L, dtype=torch.bool, device=dev)
        pm[:, :P] = True
        cond = model.branch.encode_condition(speech_condition=sc, mask=mask,
                                             cutoff_hz=cutoff, prompt_mask=pm)
        sel = torch.zeros(L, 100, dtype=torch.bool, device=dev)
        sel[P:, mask[0].flatten() > 0.99] = True
        selL = torch.zeros_like(sel)
        selL[P:, mask[0].flatten() < 0.01] = True
        refm = ref[0]

        def branch(t, y):
            t = torch.tensor(t, device=dev) if not torch.is_tensor(t) else t
            h = model._memory(t=t, y=y, text_condition=tc, speech_condition=sc,
                              padding_mask=pad)
            return model.branch(t=t, xt=y, text_condition=tc, speech_condition=sc,
                                mask=mask, cutoff_hz=cutoff, prompt_mask=pm,
                                memory=h, padding_mask=pad, condition=cond)

        noise = torch.randn_like(ref)
        # teacher-forced endpoint fidelity
        r = []
        for t in TS:
            xt = ref * t + noise * (1 - t)
            r.append(cmp(branch(t, xt)[0], refm, sel))
        R["tf"].append(r)
        # state swap at 0.6: high band of the state from the time-rolled reference
        t = 0.6
        rolled = ref.clone()
        rolled[:, P:] = torch.roll(rf, Tt // 2, 0)[None]
        xt = ref * t + noise * (1 - t)
        ys = model._project(xt, rolled * t + noise * (1 - t), mask)
        o = branch(t, ys)[0]
        R["swap"].append(cmp(o, refm, sel)[1::2] + cmp(o, rolled[0], sel)[1::2])

        # 16-step solve, 4 seeds, eval settings
        ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=16, t_shift=0.5, device=dev)
        outs, steps = [], []
        for seed in range(4):
            g = torch.Generator(device=dev).manual_seed(seed)
            x = torch.randn(ref.shape, device=dev, generator=g)
            y = x
            st = []
            for k in range(16):
                tk, dt = ts[k], ts[k + 1] - ts[k]
                v, _ = model._backbone_step(t=tk, x=x, text_condition=tc,
                                            speech_condition=sc, padding_mask=pad,
                                            guidance_scale=1.0)
                b1 = branch(tk, y)
                st.append(ratio_only(b1[0], refm, sel))
                y = model._project(x + v * dt, y + endpoint_to_velocity(b1, y, tk) * dt, mask)
                x = x + v * dt
            outs.append(model._project(x, y, mask)[0])
            steps.append(st)
        R["steps"].append(np.mean(steps, 0))
        R["solve"].append(np.mean([ratio_only(o, refm, sel) for o in outs], 0))
        O = torch.stack(outs)
        seed_std = O.std(0)
        tstd = O[:, P:].std(1, keepdim=True).mean(0)  # per-bin temporal std
        tstd = tstd.expand(L - P, -1)
        R["div"].append([(seed_std[P:][sel[P:]] / tstd[sel[P:]]).mean().item(),
                         (seed_std[P:][selL[P:]] / tstd[selL[P:]]).mean().item()])
        # share of the output's spectral detail that is a fixed per-bin pattern
        # (the same ripple every frame) rather than moving with the speech
        def static_share(m):
            d = detail(m)[1][P:][:, sel[P]]
            return (d.mean(0).pow(2).mean() / d.pow(2).mean()).item()
        R.setdefault("static", []).append(
            [np.mean([static_share(o) for o in outs]), static_share(refm)])
        R["lvl"].append([(O[:, P:][:, sel[P:]].mean() - refm[sel].mean()).item() * 10 * 4.343])

        # backbone alone, clean 24k prompt, full band
        scc = F.pad(cf[None], (0, 0, 0, L - P)) if cf.size(0) == P else None
        if scc is not None:
            g = torch.Generator(device=dev).manual_seed(0)
            x = torch.randn(ref.shape, device=dev, generator=g)
            for k in range(16):
                tk, dt = ts[k], ts[k + 1] - ts[k]
                v, _ = model._backbone_step(t=tk, x=x, text_condition=tc,
                                            speech_condition=scc, padding_mask=pad,
                                            guidance_scale=1.0)
                x = x + v * dt
            R["bb"].append(ratio_only(x[0], refm, sel))
            refc = torch.cat([cf, rf])[None]
            r = []
            for t in TS:
                xt = refc * t + noise * (1 - t)
                v, _ = model._backbone_step(t=torch.tensor(t, device=dev), x=xt,
                                            text_condition=tc, speech_condition=scc,
                                            padding_mask=pad, guidance_scale=0.0)
                r.append(cmp((xt + (1 - t) * v)[0], refm, sel))
            R["bbtf"].append(r)
        print(".", end="", flush=True)

print(f"\n{ckpt}  items={N_ITEMS}")
print("detail columns: time-ratio time-corr freq-ratio freq-corr (1,1,1,1 = reference)")
tf, bbtf = np.mean(R["tf"], 0), np.mean(R["bbtf"], 0)
for i, t in enumerate(TS):
    print(f"  tf t={t:4.2f} branch " + " ".join(f"{v:6.3f}" for v in tf[i])
          + "   backbone(clean) " + " ".join(f"{v:6.3f}" for v in bbtf[i]))
s = np.mean(R["swap"], 0)
print(f"  swap t=0.6 corr to ref time/freq {s[0]:.3f} {s[1]:.3f}  to swapped-state {s[2]:.3f} {s[3]:.3f}")
print("  solve output ratio time/freq " + " ".join(f"{v:.3f}" for v in np.mean(R["solve"], 0))
      + "   backbone(clean) solve " + " ".join(f"{v:.3f}" for v in np.mean(R["bb"], 0)))
print(f"  solve high-band mean level vs ref {np.mean(R['lvl']):+.1f} dB (log-mean)")
d = np.mean(R["div"], 0)
print(f"  seed diversity (seed std / temporal std): high {d[0]:.3f}  low(backbone) {d[1]:.3f}")
ss = np.mean(R["static"], 0)
print(f"  static share of spectral detail: output {ss[0]:.3f}  reference {ss[1]:.3f}")
st = np.mean(R["steps"], 0)
print("  per-step endpoint time-ratio: " + " ".join(f"{v[0]:.2f}" for v in st))
print("  per-step endpoint freq-ratio: " + " ".join(f"{v[1]:.2f}" for v in st))
