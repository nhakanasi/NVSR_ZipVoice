# Control 2: AudioSR on 8 kHz natural references, then both verifiers against the clean prompts.
export PYTHONUTF8=1
for c in libritts:bwe_eval vctk:vctk_bwe_eval; do n=${c%%:*}; d=${c##*:}
  echo "== audiosr $n $(date +%T)"
  /c/zipvoice_data/venv_audiosr/Scripts/python -u "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice/local/audiosr_restore.py" --test-list C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/control2/$n/refs_sr8000.tsv     --out-list C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/control2/$n/refs_audiosr_sr8000.tsv --prompt-dir C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/control2/$n/audiosr_sr8000     --timing-path C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/control2/$n/audiosr_time.tsv || echo "FAILED audiosr $n"
done
export PYTHONPATH="$(pwd -W)"
for c in libritts:bwe_eval vctk:vctk_bwe_eval; do n=${c%%:*}; d=${c##*:}
  W=C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/control2/$n/audiosr_sr8000; L=C:/zipvoice_data/data/$d/test_score.tsv
  python -u -m zipvoice.eval.speaker_similarity.sim --wav-path $W --test-list $L     --model-dir C:/zipvoice_data/download/tts_eval_models --score-path $W/sim_scores.tsv 2>&1 | grep -a "SIM-o score"
  python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/campp_sim.py $L campp_scores.tsv $W
done
echo "== all done $(date +%T)"
