"""EER companion in Carbonneau et al.'s format, on LibriTTS dev-clean.

Per speaker, 20 utterances of 4-10 s are drawn (seed 0) and split at random into a genuine
half G and a perturbed half P of 10 each. G is left untouched. Each condition rewrites every
file in P; "clean" leaves P untouched, which is their random-split control (expected EER 0.50).
Conditions:
  clean      P as recorded
  emph       1 - 0.97 z^-1, RMS matched back to the original
  deemph     1 / (1 - 0.97 z^-1), RMS matched back to the original
  sr16000    cut to 16 kHz and back, as control 1
  sr8000     cut to 8 kHz and back, as control 1
  nvsr8000   sr8000 restored by NVSR (written later by nvsr_restore.py, from in_sr8000/)
usage: eer_prep.py SCRATCH"""
import glob
import random
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

S = Path(sys.argv[1]) / "eer"
random.seed(0)
by_spk = {}
for f in sorted(glob.glob("C:/zipvoice_data/download/LibriTTS/dev-clean/*/*/*.wav")):
    f = f.replace("\\", "/")
    info = torchaudio.info(f)
    if 4 <= info.num_frames / info.sample_rate <= 10:
        by_spk.setdefault(f.rsplit("/", 1)[1].split("_")[0], []).append(f)
dirs = {c: S / c for c in ["G", "clean", "emph", "deemph", "sr16000", "sr8000", "in_sr8000"]}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)
rows, split = [], open(S / "split.tsv", "w", encoding="utf-8", newline="\n")
for spk, files in sorted(by_spk.items()):
    if len(files) < 20:
        print("skip", spk, len(files))
        continue
    pick = random.sample(files, 20)
    for i, f in enumerate(pick):
        name = f.rsplit("/", 1)[1]
        split.write(f"{spk}\t{'G' if i < 10 else 'P'}\t{name}\n")
        w, sr = torchaudio.load(f)
        assert sr == 24000
        if i < 10:
            torchaudio.save(str(dirs["G"] / name), w, sr)
            continue
        rms = w.pow(2).mean().sqrt()
        fit = lambda y: y * rms / y.pow(2).mean().sqrt()
        torchaudio.save(str(dirs["clean"] / name), w, sr)
        torchaudio.save(str(dirs["emph"] / name), fit(F.lfilter(w, torch.tensor([1.0, 0.0]), torch.tensor([1.0, -0.97]), clamp=False)), sr)
        torchaudio.save(str(dirs["deemph"] / name), fit(F.lfilter(w, torch.tensor([1.0, -0.97]), torch.tensor([1.0, 0.0]), clamp=False)), sr)
        for rate in [16000, 8000]:
            low = F.resample(w, sr, rate)
            torchaudio.save(str(dirs[f"sr{rate}"] / name), F.resample(low, rate, sr)[:, : w.size(1)], sr)
        torchaudio.save(str(dirs["in_sr8000"] / name), low, 8000)
        rows.append(f"{name[:-4]}\t-\t{dirs['in_sr8000'] / name}\t-\n")
open(S / "nvsr_in.tsv", "w", encoding="utf-8", newline="\n").writelines(rows)
print(f"{sum(len(v) >= 20 for v in by_spk.values())} speakers, {len(rows)} perturbed files per condition")
