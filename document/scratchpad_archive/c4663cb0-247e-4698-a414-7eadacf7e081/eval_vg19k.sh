# Score checkpoint 19000 of the VGG run (18000-19000 averaged), then resume
# the run to its 20000 gate.
E=/c/zipvoice_data/exp
S="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
W="/g/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
cd "$W"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}" PYTHONUTF8=1
G=$E/adapter_loravg_eval19k.log
python -m zipvoice.bin.generate_averaged_model --iter 19000 --avg 1 --model-name zipvoice_adapter --exp-dir $E/adapter_lora_vg > $G 2>&1
# Stage 2 loads iter-N-avg-2.pt by name; this one is the 18000-19000 average.
cp $E/adapter_lora_vg/iter-19000-avg-1.pt $E/adapter_lora_vg/iter-19000-avg-2.pt || { echo EVAL_FAILED; exit 1; }
stage=2 stop_stage=3 system=adapter_loravg19k exp_dir=$E/adapter_lora_vg num_iters=19000 eval_rates="8000 16000" bash run_adapter_libritts.sh >> $G 2>&1
echo "$(date +%T) EVAL_DONE 19000"
grep -E "SIM-o score|WER: [0-9]|UTMOS score:" $G | sed -E 's/.*(SIM-o score: [0-9.]+|WER: [0-9.%]+|UTMOS score: [0-9.]+).*/GATE \1/'
python "$(cygpath -w "$S/hfband.py")" adapter_loravg19k_sr8000 adapter_loravg19k_sr16000 | sed "s/^/HF /"
python "$(cygpath -w "$S/lpsim.py")" adapter_loravg19k_sr8000 adapter_loravg19k_sr16000 2>/dev/null | sed "s/^/GATE /"
for s in adapter_loravg19k_sr8000 adapter_loravg19k_sr16000; do
  python -m zipvoice.eval.speaker_similarity.sim --wav-path C:/zipvoice_data/exp/results/${s}_lp4k --test-list C:/zipvoice_data/data/bwe_eval/test_score.tsv --model-dir C:/zipvoice_data/download/tts_eval_models --score-path C:/zipvoice_data/exp/results/${s}_lp4k/sim_scores.tsv 2>&1 | grep "SIM-o score" | sed "s/.*SIM-o/GATE $s lp4k SIM-o/"
done
echo "SCORE_DONE; resuming"
bash "$S/lora_supervise_vg2.sh"
