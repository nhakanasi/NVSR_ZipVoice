import json, sys, torch
sys.path.insert(0, ".")
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter, adapter_kwargs

cfg = json.load(open("egs/zipvoice/conf/zipvoice_adapter.json"))
m = cfg["model"]
# Shrink the backbone so this runs on CPU in seconds.
for k in list(m):
    if k.startswith("fm_decoder") and isinstance(m[k], list):
        m[k] = [min(v, 1) if "num_layers" in k else v for v in m[k]]
kw = adapter_kwargs(cfg, steer_all_bins=True)
model = ZipVoiceAdapter(**m, **kw, vocab_size=50, pad_id=0)
model.set_trainable("adapter")
torch.manual_seed(0)

B, T, F = 2, 60, model.feat_dim
tokens = [[1, 2, 3, 4], [5, 6, 7]]
features = torch.randn(B, T, F) * 0.3
lens = torch.tensor([60, 50])
noise = torch.randn(B, T, F)
t = torch.rand(B, 1, 1)
cut = torch.tensor([4000.0, 4000.0])

# 1. Inert on a full-band prompt, writes low bins otherwise.
for p in model.adapter.parameters():
    p.data.normal_(0, 0.05)
model.eval()
with torch.no_grad():
    xt = noise
    text = torch.randn(B, T, m.get("text_encoder_dim", 192) if False else F)
fm, rec, aux = model(tokens, features, lens, noise, t, cutoff_hz=cut)
r = model._last_residual
_, band_mask = model._band_limit(features, cut, randomize=False)
low = (1 - band_mask) > 0.99
print("low-band residual rms", r[..., low[0, 0]].pow(2).mean().sqrt().item())
fm, rec, aux = model(tokens, features, lens, noise, t,
                     cutoff_hz=torch.tensor([24000.0, 24000.0]))
print("full-band residual max", model._last_residual.abs().max().item())

# 2. Rollout with K=3 and reference.
model.train()
for K in (1, 3):
    model.zero_grad()
    fm, rec, aux = model(
        tokens, features, lens, noise, t, cutoff_hz=cut,
        identity_rollout_steps=6, identity_rollout_prob=1.0,
        identity_rollout_grad_steps=K, identity_rollout_reference=True, identity_rollout_distance="l2",
    )
    assert aux["is_rollout"]
    loss = aux["rollout_low_loss"] + aux["rollout_high_loss"]
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.adapter.parameters() if p.grad is not None)
    bb = [n for n, p in model.fm_decoder.named_parameters() if p.grad is not None]
    print(f"K={K} low {aux['rollout_low_loss'].item():.4f} high {aux['rollout_high_loss'].item():.4f} "
          f"adapter grad {g:.4g} backbone grads {len(bb)} detach reset {model._detach_backbone}")
