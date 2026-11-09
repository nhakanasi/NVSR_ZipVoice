import glob, torch, torchaudio
from zipvoice.models.modules.bwe import resample_band_limit
from zipvoice.utils.feature import VocosFbank
fb = VocosFbank().fbank
f = sorted(glob.glob("C:/zipvoice_data/data/bwe_eval/prompts/clean/*.wav"))[0]
w, sr = torchaudio.load(f); w = torchaudio.functional.resample(w, sr, 24000)
T = w.size(1) // 256 + 1
hz = torchaudio.functional.melscale_fbanks(513, 0, 12000, 100, 24000, norm=None, mel_scale="htk").argmax(0) * 24000 / 1024
bands = [(3e3, 4e3), (4e3, 6e3), (6e3, 8e3), (8e3, 12e3)]
clean = fb(w).clamp(min=1e-7).log().transpose(1, 2)[0]
def rep(tag, m):
    d = (m.mean(0) - clean.mean(0)) * 8.686
    print(tag, " ".join(f"{float(d[(hz>=a)&(hz<b)].mean()):7.1f}" for a, b in bands))
for c in (4000., 8000.):
    kw = dict(audio=w, audio_lens=torch.tensor([w.size(1)]), cutoff_hz=torch.tensor([c]), fbank=fb, num_frames=T)
    rep(f"{c:.0f} exact  ", resample_band_limit(**kw)[0])
    for i in range(6):
        torch.manual_seed(i); rep(f"{c:.0f} var{i}   ", resample_band_limit(variants=True, **kw)[0])
