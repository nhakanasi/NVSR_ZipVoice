#!/bin/bash
# Stage 11 is stopped at checkpoint-13000. Score steps 13000 and 10000, resume.
set -u
EXP=/c/zipvoice_data/exp/adapter_steer
LOG=/c/zipvoice_data/exp/adapter_steer.log
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
# Windows python splits PYTHONPATH on ';'.
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"

for it in 13000 10000; do
  python -m zipvoice.bin.generate_averaged_model --iter $it --avg 2 \
    --model-name zipvoice_adapter --exp-dir $EXP > /c/zipvoice_data/exp/steer_avg_$it.log 2>&1 \
    || { echo "[pause] averaging $it failed"; tail -3 /c/zipvoice_data/exp/steer_avg_$it.log; continue; }
  echo "[pause] scoring $it $(date +%T)"
  stage=2 stop_stage=3 system=adapter_steer${it%000}k exp_dir=$EXP num_iters=$it project_low_band=0 \
    bash run_adapter_libritts.sh > /c/zipvoice_data/exp/steer_eval_$it.log 2>&1
  echo "[pause] scored $it exit $? $(date +%T)"
done

echo "[pause] resuming training $(date +%T)"
stage=11 stop_stage=11 steer_resume=$EXP/checkpoint-13000.pt \
  bash run_adapter_libritts.sh >> $LOG 2>&1
echo "[pause] training exit $? $(date +%T)"
