# Restart the LoRA trainer at every checkpoint to reset its host-memory growth;
# every 2000 steps, score 8k/16k before resuming.
E=/c/zipvoice_data/exp
L=$E/adapter_lora_rs.log
S="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
W="/g/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
PS1FILE="$(cygpath -w "C:\Users\Admin\AppData\Local\Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/stop_trainer.ps1")"
cd "$W"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}" PYTHONUTF8=1
export curriculum_rho=0 lora_num_iters=26000 lora_rollout_cutoffs="4000 8000" lora_band_level_weights="0.1 0.05" lora_band_level_underfill=true lora_resample_degrade=true lora_degrade_variants=true lora_exp_dir=$E/adapter_lora_rs
lora_resume=$E/adapter_lora/checkpoint-18000.pt stage=4 stop_stage=4 bash run_adapter_libritts.sh >> $L 2>&1 &
echo "$(date +%T) finetune from 18000 with resampled prompts and filter variants"
for N in $(seq 19000 1000 26000); do
  nxt=$((N/10+1))
  until [ -f $E/adapter_lora_rs/checkpoint-$N.pt ] && grep -qE "global_batch_idx: ${nxt}[0-9]," $L; do
    sleep 20
    if tail -n 50 $L | grep -qE 'Traceback|out of memory'; then echo "TRAIN_FAILED near $N"; exit 1; fi
  done
  echo "$(date +%T) ckpt $N: $(powershell -NoProfile -ExecutionPolicy Bypass -File "$PS1FILE")"
  if [ $((N % 2000)) -eq 0 ]; then
    python -m zipvoice.bin.generate_averaged_model --iter $N --avg 2 --model-name zipvoice_adapter --exp-dir $E/adapter_lora_rs > $E/adapter_lorars_eval$((N/1000))k.log 2>&1
    stage=2 stop_stage=3 system=adapter_lorars$((N/1000))k exp_dir=$E/adapter_lora_rs num_iters=$N eval_rates="8000 16000" bash run_adapter_libritts.sh >> $E/adapter_lorars_eval$((N/1000))k.log 2>&1 ; echo "$(date +%T) EVAL_DONE $N"; python "$(cygpath -w "$S/hfband.py")" adapter_lorars$((N/1000))k_sr8000 adapter_lorars$((N/1000))k_sr16000 | sed "s/^/HF /"
  fi
  lora_resume=$E/adapter_lora_rs/checkpoint-$N.pt stage=4 stop_stage=4 bash run_adapter_libritts.sh >> $L 2>&1 &
  echo "$(date +%T) resumed from $N"
done
wait
