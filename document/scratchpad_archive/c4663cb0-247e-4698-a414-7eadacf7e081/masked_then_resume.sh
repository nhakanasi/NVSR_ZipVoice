#!/bin/bash
# Score Stage 11's step-10000 weights with the write masked (Stage 12 at step 0),
# then resume Stage 12 from checkpoint-2000 to 3000 and score that.
set -u
EXP=/c/zipvoice_data/exp/adapter_hbfix
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
echo "[masked] scoring $(date +%T)"
stage=2 stop_stage=3 system=adapter_steermasked10k exp_dir=/c/zipvoice_data/exp/adapter_steer_masked \
  num_iters=10000 project_low_band=0 bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_adapter_steermasked10k.log 2>&1
echo "[masked] scored exit $? $(date +%T)"
echo "[hbfix] resuming $(date +%T)"
stage=12 stop_stage=12 hbfix_resume=$EXP/checkpoint-2000.pt bash run_adapter_libritts.sh >> /c/zipvoice_data/exp/adapter_hbfix.log 2>&1
echo "[hbfix] training exit $? $(date +%T)"
# Stage 12 writes iter-3000-avg-2.pt itself.
echo "[hbfix] scoring 3000 $(date +%T)"
stage=2 stop_stage=3 system=adapter_hbfix3k exp_dir=$EXP num_iters=3000 project_low_band=0 \
  bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_adapter_hbfix3k.log 2>&1
echo "[hbfix] scored 3000 exit $? $(date +%T)"
