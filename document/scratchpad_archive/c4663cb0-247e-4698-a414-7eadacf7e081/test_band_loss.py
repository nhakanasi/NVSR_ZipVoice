import torch
from zipvoice.bin.train_zipvoice_dual import band_level_loss
from zipvoice.models.modules.bwe import band_mask
from zipvoice.utils.common import AttributeDict

torch.manual_seed(0)
B, T, F = 3, 50, 100
centers = torch.linspace(0, 11800, F)
cutoff = torch.tensor([4000.0, 8000.0, 12000.0])
mask = band_mask(cutoff, centers, F)
features = torch.randn(B, T, F)
frame_mask = torch.ones(B, T, dtype=torch.bool)
frame_mask[:, :10] = False
params = AttributeDict(band_t_min=0.5)
t = torch.tensor([0.9, 0.9, 0.9]).view(B, 1, 1)

def run(x1_hat):
    x1_hat = x1_hat.clone().requires_grad_(True)
    terms = dict(t=t, frame_mask=frame_mask, band_mask=mask, x1_hat=x1_hat)
    loss = band_level_loss(params, terms, features, centers)
    loss.backward()
    return loss.item(), x1_hat.grad

l0, _ = run(features)
print("identical", l0)
l1, g = run(features - 1.0)
print("shift -1", l1, "grad below cutoff utt0", g[0, :, :30].abs().sum().item(),
      "grad above", g[0, :, 60:].abs().sum().item(), "grad prompt frames", g[:, :10].abs().sum().item(),
      "grad full-band utt2", g[2].abs().sum().item())
params.band_t_min = 0.95
print("all gated", run(features - 1.0)[0])
