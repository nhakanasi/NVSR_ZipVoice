# NVSR prompt cascade resynthesis on CPU (original wavs not kept), then both verifiers.
export PYTHONPATH="$(pwd -W)" CUDA_VISIBLE_DEVICES="" PYTHONUTF8=1
for c in libritts:bwe_eval vctk:vctk_bwe_eval; do n=${c%%:*}; d=${c##*:}
  W=C:/zipvoice_data/exp/results_rs/$n/cascade_nvsr_sr8000; L=C:/zipvoice_data/data/$d/test_score.tsv
  echo "== $n $(date +%T)"
  python -u -m zipvoice.bin.infer_zipvoice --model-name zipvoice --tokenizer emilia     --test-list C:/zipvoice_data/data/$d/test_nvsr8000.tsv --res-dir $W --num-step 16 --seed 666 > $W.log 2>&1 || echo "FAILED $n"
  python -u -m zipvoice.eval.speaker_similarity.sim --wav-path $W --test-list $L     --model-dir C:/zipvoice_data/download/tts_eval_models --score-path $W/sim_scores.tsv 2>&1 | grep -a "SIM-o score"
  python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/campp_sim.py $L campp_scores.tsv $W 2>&1 | grep -av -i warn
done
echo "== all done $(date +%T)"
