import json, torch
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter
c = json.load(open("egs/zipvoice/conf/zipvoice_adapter.json"))
c["model"].update(fm_decoder_num_layers=[1,1,1,1,1])
torch.manual_seed(0)
stock = ZipVoiceAdapter(**c["model"], **c["adapter"], vocab_size=100, pad_id=0)
torch.manual_seed(0)
lora = ZipVoiceAdapter(**c["model"], **c["adapter"], lora_rank=8, vocab_size=100, pad_id=0)
missing, unexpected = lora.load_state_dict(stock.state_dict(), strict=False)
assert not unexpected and all(k.endswith(("lora_A", "lora_B")) or k.startswith("prev_residual_proj") for k in missing), missing[:3]
print("lora tensors", len(missing), sum(p.numel() for n, p in lora.named_parameters() if "lora_" in n))
# make LoRA large so any leak would show
for m in lora._lora_modules:
    m.lora_B.data.normal_(std=0.1)
stock.eval(); lora.eval()
B, T, F = 3, 40, 100
x = torch.randn(B, T, F); text = torch.randn(B, T, F); speech = torch.randn(B, T, F)
pad = torch.zeros(B, T, dtype=torch.bool)
mask = torch.zeros(B, 1, F); mask[1, :, 60:] = 1.0  # only utt 1 band-limited
t = torch.full((B, 1, 1), 0.3)
kw = dict(t=t, xt=x, text_condition=text, speech_condition=speech, padding_mask=pad,
          band_mask=mask, cutoff_hz=torch.tensor([12000., 4000., 12000.]),
          guidance_scale=None)
with torch.no_grad():
    a = stock.forward_fm_decoder(**kw); b = lora.forward_fm_decoder(**kw)
d = (a - b).abs().amax(dim=(1, 2))
print("per-utt max diff", d.tolist())
assert d[0] == 0 and d[2] == 0 and d[1] > 0
# gate cleared afterwards
assert all(m._lora_gate is None for m in lora._lora_modules)
# CFG-style doubled batch
kw2 = dict(kw, t=torch.full((2*B,1,1),0.3), xt=torch.cat([x,x]), text_condition=torch.cat([text,text]),
           speech_condition=torch.cat([speech,speech]), padding_mask=torch.cat([pad,pad]))
with torch.no_grad(): lora.forward_fm_decoder(**kw2)
# rollout reaches LoRA
lora.train(); lora.set_trainable("lora")
assert not lora._released_decoder_trainable()
x1, idx, ref = lora.identity_rollout(torch.randn(B, T, F), text, speech, pad, mask,
    torch.tensor([12000., 4000., 12000.]), torch.ones(B, T, dtype=torch.bool),
    num_step=4, grad_steps=2, clean_condition=speech)
((x1 - ref) ** 2).mean().backward()
g = [m.lora_B.grad for m in lora._lora_modules]
print("idx", idx, "lora_B with grad", sum(x is not None and x.abs().sum() > 0 for x in g), "/", len(g))
print("released decoder grads", sum(p.grad is not None for n,p in lora.fm_decoder.named_parameters() if "lora_" not in n))
print("prev_residual_proj grad", lora.prev_residual_proj.weight.grad.abs().sum().item())
# carried solve: full-band utterances still stock at every step
for m in lora._lora_modules: m.lora_B.grad = None
lora.eval(); stock.eval()
with torch.no_grad():
    lora._carry_residual = True; lora._prev_residual = None
    xa = xb = torch.randn(B, T, F)
    for k in range(4):
        tk = torch.full((B, 1, 1), k / 4)
        va = stock.forward_fm_decoder(**dict(kw, t=tk, xt=xa)); vb = lora.forward_fm_decoder(**dict(kw, t=tk, xt=xb))
        xa = xa + va / 4; xb = xb + vb / 4
    lora._carry_residual = False
    print("prev set", lora._prev_residual is not None)
    d = (xa - xb).abs().amax(dim=(1, 2)); print("solve per-utt diff", d.tolist())
    assert d[0] == 0 and d[2] == 0
