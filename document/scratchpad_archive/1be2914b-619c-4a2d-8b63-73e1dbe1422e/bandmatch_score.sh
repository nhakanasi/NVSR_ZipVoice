#!/bin/bash
# Band-matched scoring, step 2: both verifiers on the low-passed outputs against the low-passed prompt.
SP="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/1be2914b-619c-4a2d-8b63-73e1dbe1422e/scratchpad"
OS="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
cd 'G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice'
export PYTHONPATH="$(pwd -W)" PYTHONUTF8=1
for n in libritts vctk; do
  L=$SP/bandmatch/$n/test_lp8.tsv
  dirs=()
  for k in n24 n8 z24 z8 f24 f8 cas asr nvsr; do W=$SP/bandmatch/$n/$k; dirs+=("$W")
    echo "== W $n $k"
    python -u -m zipvoice.eval.speaker_similarity.sim --wav-path $W --test-list $L --model-dir C:/zipvoice_data/download/tts_eval_models --score-path $W/sim_scores.tsv 2>&1 | grep -a "SIM-o score"
  done
  echo "== C $n"
  python -u $OS/campp_sim.py $L campp_scores.tsv "${dirs[@]}" 2>&1 | grep -av -i warn
done
echo SCOREDONE
