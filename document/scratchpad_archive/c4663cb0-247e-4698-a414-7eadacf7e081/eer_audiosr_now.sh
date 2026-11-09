echo "== starting AudioSR now $(date +%T)"
export PYTHONUTF8=1
/c/zipvoice_data/venv_audiosr/Scripts/python -u "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice/local/audiosr_restore.py" --test-list C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/nvsr_in.tsv --out-list C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/audiosr_out.tsv --prompt-dir C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/audiosr_raw --timing-path C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer/audiosr_time.tsv 2>&1 | grep -a "Restored\|Error\|error"
export PYTHONPATH="$(pwd -W)"
python -u - <<'PY'
import torchaudio
from pathlib import Path
S = Path("C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer"); (S / "audiosr8000").mkdir(exist_ok=True); d = []
for f in sorted((S / "clean").glob("*.wav")):
    w, sr = torchaudio.load(str(S / "audiosr_raw" / f.name)); n = torchaudio.info(str(f)).num_frames
    assert sr == 24000 and w.size(1) >= n, (f.name, sr, w.size(1), n)
    d.append((w.size(1) - n) / sr); torchaudio.save(str(S / "audiosr8000" / f.name), w[:, :n], sr)
print(f"audiosr padding trimmed, s: median {sorted(d)[len(d) // 2]:.2f} max {max(d):.2f}", flush=True)
PY
python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer_score.py C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad clean emph deemph noise40 noise20 noise0 sr16000 sr8000 nvsr8000 audiosr8000 2>&1 | grep -a "EER\|Error\|error"
python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/eer_paired.py C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad nvsr8000:sr8000 audiosr8000:sr8000 sr8000:deemph sr8000:noise20 noise0:sr8000
echo "== all done $(date +%T)"
