# Score resynthesised stock with both verifiers on CPU, then decompose.
export PYTHONPATH="$(pwd -W)" CUDA_VISIBLE_DEVICES="" PYTHONUTF8=1
for c in libritts:bwe_eval vctk:vctk_bwe_eval; do n=${c%%:*}; d=${c##*:}
  L=C:/zipvoice_data/data/$d/test_score.tsv
  for rate in 8000 24000; do W=C:/zipvoice_data/exp/results_rs/$n/stock_sr$rate
    python -u -m zipvoice.eval.speaker_similarity.sim --wav-path $W --test-list $L       --model-dir C:/zipvoice_data/download/tts_eval_models --score-path $W/sim_scores.tsv 2>&1 | grep -a "SIM-o score"
  done
  python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/campp_sim.py $L campp_scores.tsv C:/zipvoice_data/exp/results_rs/$n/stock_sr8000 C:/zipvoice_data/exp/results_rs/$n/stock_sr24000
done
python -u C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/decompose.py C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad
