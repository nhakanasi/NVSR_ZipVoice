# Restart the LoRA trainer at every checkpoint to reset its host-memory growth;
# every 2000 steps, score 8k/16k before resuming.
E=/c/zipvoice_data/exp
L=$E/adapter_lora.log
W="/g/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
PS1FILE="$(cygpath -w "C:\Users\Admin\AppData\Local\Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad/stop_trainer.ps1")"
cd "$W"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}" PYTHONUTF8=1
for N in $(seq 5000 1000 19000); do
  nxt=$((N/10+1))
  until [ -f $E/adapter_lora/checkpoint-$N.pt ] && grep -qE "global_batch_idx: ${nxt}[0-9]," $L; do
    sleep 20
    if tail -n 50 $L | grep -qE 'Traceback|out of memory'; then echo "TRAIN_FAILED near $N"; exit 1; fi
  done
  echo "$(date +%T) ckpt $N: $(powershell -NoProfile -ExecutionPolicy Bypass -File "$PS1FILE")"
  if [ $((N % 2000)) -eq 0 ]; then
    python -m zipvoice.bin.generate_averaged_model --iter $N --avg 2 --model-name zipvoice_adapter --exp-dir $E/adapter_lora > $E/adapter_lora_eval$((N/1000))k.log 2>&1
    stage=2 stop_stage=3 system=adapter_lora$((N/1000))k exp_dir=$E/adapter_lora num_iters=$N eval_rates="8000 16000" bash run_adapter_libritts.sh >> $E/adapter_lora_eval$((N/1000))k.log 2>&1 && echo "$(date +%T) EVAL_DONE $N" || echo "EVAL_FAILED $N"
  fi
  lora_resume=$E/adapter_lora/checkpoint-$N.pt stage=4 stop_stage=4 bash run_adapter_libritts.sh >> $L 2>&1 &
  echo "$(date +%T) resumed from $N"
done
wait
