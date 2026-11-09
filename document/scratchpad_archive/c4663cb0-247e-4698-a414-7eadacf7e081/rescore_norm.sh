# Rescore both verifiers on the loudness-normalised listening-test conditions (CPU).
export PYTHONPATH="$(pwd -W)" CUDA_VISIBLE_DEVICES="" PYTHONUTF8=1
S=C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad
python -u "$S/norm_copy.py" $S
L=$S/norm/test_score.tsv
for c in nat24 nat8 audiosr nvsr stock24 stock8 cascade; do
  python -u -m zipvoice.eval.speaker_similarity.sim --wav-path $S/norm/$c --test-list $L \
    --model-dir C:/zipvoice_data/download/tts_eval_models --score-path $S/norm/$c/sim_scores.tsv 2>&1 | grep -a "SIM-o score" | sed "s/^/$c /"
done
python -u "$S/campp_sim.py" $L campp_scores.tsv $S/norm/nat24 $S/norm/nat8 $S/norm/audiosr $S/norm/nvsr \
  $S/norm/stock24 $S/norm/stock8 $S/norm/cascade
echo RESCORE_DONE
