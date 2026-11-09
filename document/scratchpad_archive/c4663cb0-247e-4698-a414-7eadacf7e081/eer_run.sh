# EER companion: NVSR on the 8 kHz perturbed halves, length fit, then both verifiers.
export PYTHONUTF8=1 PYTHONPATH="$(pwd -W)"
python -u "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice/local/nvsr_restore.py" --test-list C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/nvsr_in.tsv --out-list C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/nvsr_out.tsv --prompt-dir C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/nvsr_raw 2>&1 | grep -a "Restored\|Error\|error" 
python -u - <<'PY'
import torch, torchaudio
from pathlib import Path
S = Path("C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer"); (S / "nvsr8000").mkdir(exist_ok=True); d = []
for f in sorted((S / "clean").glob("*.wav")):
    w, sr = torchaudio.load(str(S / "nvsr_raw" / f.name)); n = torchaudio.info(str(f)).num_frames
    d.append(w.size(1) - n); torchaudio.save(str(S / "nvsr8000" / f.name), torch.nn.functional.pad(w, (0, max(0, n - w.size(1))))[:, :n], sr)
print("nvsr length minus reference, samples:", min(d), max(d), flush=True)
PY
python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer_score.py C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad clean emph deemph sr16000 sr8000 nvsr8000 2>&1 | grep -a "EER\|Error\|error"
echo "== all done $(date +%T)"
