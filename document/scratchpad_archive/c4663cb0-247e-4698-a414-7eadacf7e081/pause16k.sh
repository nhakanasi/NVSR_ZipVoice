#!/bin/bash
# Stop stage 11 once checkpoint-16000 is written and validated, score it,
# resume to 20000, score that, then run and score the rollout-off control.
set -u
EXP=/c/zipvoice_data/exp/adapter_steer
CTRL=/c/zipvoice_data/exp/adapter_steer_ctrl
LOG=/c/zipvoice_data/exp/adapter_steer.log
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"

until grep -q "global_batch_idx: 16000, validation" $LOG; do sleep 20; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match 'train_zipvoice_adapter' } | % { taskkill /F /T /PID \$_.ProcessId }"
sleep 10
echo "[q] stopped at 16000 $(date +%T)"

avg_score () {  # exp_dir iter system
  python -m zipvoice.bin.generate_averaged_model --iter $2 --avg 2 \
    --model-name zipvoice_adapter --exp-dir $1 > /c/zipvoice_data/exp/avg_$3.log 2>&1 \
    || { echo "[q] averaging $3 failed"; tail -3 /c/zipvoice_data/exp/avg_$3.log; return 1; }
  echo "[q] scoring $3 $(date +%T)"
  stage=2 stop_stage=3 system=$3 exp_dir=$1 num_iters=$2 project_low_band=0 \
    bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_$3.log 2>&1
  echo "[q] scored $3 exit $? $(date +%T)"
}

avg_score $EXP 16000 adapter_steer16k

echo "[q] resuming to 20000 $(date +%T)"
stage=11 stop_stage=11 steer_resume=$EXP/checkpoint-16000.pt \
  bash run_adapter_libritts.sh >> $LOG 2>&1
echo "[q] training exit $? $(date +%T)"
[ -f $EXP/checkpoint-20000.pt ] && avg_score $EXP 20000 adapter_steer20k

echo "[q] control training $(date +%T)"
stage=11 stop_stage=11 steer_exp_dir=$CTRL steer_resume=$EXP/checkpoint-10000.pt \
  steer_rollout_prob=0 steer_num_iters=13000 \
  bash run_adapter_libritts.sh > /c/zipvoice_data/exp/adapter_steer_ctrl.log 2>&1
echo "[q] control training exit $? $(date +%T)"
[ -f $CTRL/checkpoint-13000.pt ] && avg_score $CTRL 13000 adapter_steerctrl13k
echo "[q] done $(date +%T)"
