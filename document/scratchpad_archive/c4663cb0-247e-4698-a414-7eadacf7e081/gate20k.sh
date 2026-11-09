# Halt the resample-degrade finetune right after its 20000 eval, so the gate
# decides whether it continues toward 26000.
O="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/tasks/bv92nyh2l.output"
until grep -q "EVAL_DONE 20000" "$O" 2>/dev/null || grep -q FAILED "$O" 2>/dev/null; do sleep 2; done
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/stop_supervisor_rs.ps1")"
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/stop_trainer.ps1")"
tail -5 "$O"
L=/c/zipvoice_data/exp/adapter_lorars_eval20k.log
grep -E "SIM-o score|WER: [0-9]|UTMOS score:" $L | sed -E 's/.*(SIM-o score: [0-9.]+|WER: [0-9.%]+|UTMOS score: [0-9.]+).*/\1/'
cd "/g/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep"
PYTHONPATH="." python "$(cygpath -w "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/hfband.py")" adapter_lorars20k_sr8000 adapter_lorars20k_sr16000
