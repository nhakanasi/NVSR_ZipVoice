import torch
from zipvoice.models.modules.bwe import resample_band_limit
from zipvoice.utils.feature import VocosFbank
fb = VocosFbank().fbank.cuda()
w = torch.randn(4, 48000, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    for i in range(20):
        torch.manual_seed(i)
        m = resample_band_limit(w, torch.full((4,), 48000, device="cuda"), torch.tensor([4000., 4000., 8000., 12000.], device="cuda"), fb, 188, variants=True)
print("ok", m.shape, torch.isfinite(m).all().item())
