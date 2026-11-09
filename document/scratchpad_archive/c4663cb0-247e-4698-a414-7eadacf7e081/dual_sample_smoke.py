"""The dual sampler with the memory taken from y: runs, and keeps the low band stock.

A fresh branch on the released backbone. Observed bins of the dual sample must be
bit-identical to a plain ZipVoice solve from the same noise, since the backbone
trajectory never reads the branch whatever the memory is computed on.
"""
import json

import torch

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_dual import ZipVoiceDual
from zipvoice.utils.checkpoint import load_checkpoint

D = "C:/zipvoice_data/download/zipvoice"
cfg = json.load(open("egs/zipvoice/conf/zipvoice_dual.json"))
dev = torch.device("cuda")
kw = dict(vocab_size=400, pad_id=0) if False else {}
tok_n = sum(1 for _ in open(f"{D}/tokens.txt", encoding="utf-8"))
dual = ZipVoiceDual(**cfg["model"], **cfg.get("dual", {}), vocab_size=tok_n, pad_id=0).to(dev).eval()
missing, unexpected = dual.load_state_dict(torch.load(f"{D}/model.pt", map_location="cpu")["model"], strict=False)
assert not unexpected and all(k.startswith("branch.") for k in missing), (unexpected, missing[:5])
stock = ZipVoice(**cfg["model"], vocab_size=tok_n, pad_id=0).to(dev).eval()
load_checkpoint(filename=f"{D}/model.pt", model=stock, strict=True)

T_p = 120
prompt = torch.randn(1, T_p, 100, device=dev) * 0.3
args = dict(tokens=[[5, 6, 7, 8, 9, 10] * 6], prompt_tokens=[[11, 12, 13] * 6],
            prompt_features=prompt, prompt_features_lens=torch.tensor([T_p], device=dev),
            num_step=16, guidance_scale=1.0, t_shift=0.5)
with torch.no_grad():
    torch.manual_seed(0)
    a, alen, _, _ = dual.sample(**args, cutoff_hz=4000.0)
    torch.manual_seed(0)
    b, blen, _, _ = stock.sample(**args)
mask = dual._band_mask(torch.tensor([4000.0], device=dev))[0, 0]
obs = mask == 0
print("shapes", tuple(a.shape), tuple(b.shape))
print("observed bins", int(obs.sum()), "max |dual-stock| observed", float((a - b)[..., obs].abs().max()))
print("missing-band mean |dual-stock|", float((a - b)[..., mask == 1].abs().mean()))
