"""Vocode each reference's own mel (ZipVoice features) with Vocos on CPU."""
import sys
from pathlib import Path
import torch, torchaudio, soundfile as sf
from zipvoice.utils.feature import VocosFbank
from zipvoice.bin.infer_zipvoice import get_vocoder
torch.set_num_threads(4)
voc = get_vocoder("C:/zipvoice_data/download/vocos-mel-24khz").eval()
fb = VocosFbank()
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for p in sorted(src.glob("*.wav")):
    w, sr = torchaudio.load(str(p))
    w = w.mean(0, keepdim=True)
    if sr != 24000: w = torchaudio.functional.resample(w, sr, 24000)
    with torch.no_grad():
        f = fb.extract(w, sampling_rate=24000)          # (T, 100) log-mel
        y = voc.decode(f.T.unsqueeze(0)).squeeze().clamp(-1, 1)
    sf.write(str(dst / p.name), y.numpy(), 24000)
