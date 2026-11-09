# Restart the LoRA trainer at every checkpoint to reset its host-memory growth;
# every 2000 steps, score 8k/16k before resuming.
E=/c/zipvoice_data/exp
L=$E/adapter_lora_rsid.log
S="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
W="/g/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
PS1FILE="$(cygpath -w "C:\Users\Admin\AppData\Local\Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/stop_trainer.ps1")"
cd "$W"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}" PYTHONUTF8=1
export curriculum_rho=0 lora_num_iters=26000 lora_rollout_cutoffs="4000 8000" lora_band_level_weight=0 lora_identity_weight=0.5 lora_identity_low_band=true lora_resample_degrade=true lora_degrade_variants=true lora_exp_dir=$E/adapter_lora_rsid
lora_resume=$E/adapter_lora/checkpoint-18000.pt stage=4 stop_stage=4 bash run_adapter_libritts.sh >> $L 2>&1 &
echo "$(date +%T) finetune from 18000 with resampled prompts and filter variants, low-band identity 0.5, no band-level; gate at 20000"
for N in $(seq 19000 1000 20000); do
  nxt=$((N/10+1))
  until [ -f $E/adapter_lora_rsid/checkpoint-$N.pt ] && grep -qE "global_batch_idx: ${nxt}[0-9]," $L; do
    sleep 20
    if tail -n 50 $L | grep -qE 'Traceback|out of memory'; then echo "TRAIN_FAILED near $N"; exit 1; fi
  done
  echo "$(date +%T) ckpt $N: $(powershell -NoProfile -ExecutionPolicy Bypass -File "$PS1FILE")"
  if [ $((N % 2000)) -eq 0 ]; then
    python -m zipvoice.bin.generate_averaged_model --iter $N --avg 2 --model-name zipvoice_adapter --exp-dir $E/adapter_lora_rsid > $E/adapter_lorarsid_eval$((N/1000))k.log 2>&1
    stage=2 stop_stage=3 system=adapter_lorarsid$((N/1000))k exp_dir=$E/adapter_lora_rsid num_iters=$N eval_rates="8000 16000" bash run_adapter_libritts.sh >> $E/adapter_lorarsid_eval$((N/1000))k.log 2>&1 ; echo "$(date +%T) EVAL_DONE $N"; python "$(cygpath -w "$S/hfband.py")" adapter_lorarsid$((N/1000))k_sr8000 adapter_lorarsid$((N/1000))k_sr16000 | sed "s/^/HF /"
  fi
  [ $N -lt 20000 ] && lora_resume=$E/adapter_lora_rsid/checkpoint-$N.pt stage=4 stop_stage=4 bash run_adapter_libritts.sh >> $L 2>&1 &
  [ $N -lt 20000 ] && echo "$(date +%T) resumed from $N"
done
# Gate: scores at 20000, plus SIM on outputs low-passed at 4 kHz, which is
# what separates low-band identity from the invented band.
E20=$E/adapter_lorarsid_eval20k.log
grep -E "SIM-o score|WER: [0-9]|UTMOS score:" $E20 | sed -E 's/.*(SIM-o score: [0-9.]+|WER: [0-9.%]+|UTMOS score: [0-9.]+).*/GATE \1/'
python "$(cygpath -w "$S/lpsim.py")" adapter_lorarsid20k_sr8000 adapter_lorarsid20k_sr16000 2>/dev/null | sed "s/^/GATE /"
for s in adapter_lorarsid20k_sr8000 adapter_lorarsid20k_sr16000; do
  python -m zipvoice.eval.speaker_similarity.sim --wav-path C:/zipvoice_data/exp/results/${s}_lp4k --test-list C:/zipvoice_data/data/bwe_eval/test_score.tsv --model-dir C:/zipvoice_data/download/tts_eval_models --score-path C:/zipvoice_data/exp/results/${s}_lp4k/sim_scores.tsv 2>&1 | grep "SIM-o score" | sed "s/.*SIM-o/GATE $s lp4k SIM-o/"
done
echo "GATE_DONE"
