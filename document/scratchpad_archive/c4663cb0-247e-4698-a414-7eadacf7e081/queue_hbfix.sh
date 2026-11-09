#!/bin/bash
# Wait for Stage 12 to finish averaging, then score step 3000 as adapter_hbfix3k.
set -u
EXP=/c/zipvoice_data/exp/adapter_hbfix
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
until ls $EXP/iter-3000-avg-2.pt >/dev/null 2>&1 || grep -q Traceback /c/zipvoice_data/exp/adapter_hbfix.log; do sleep 60; done
grep -q Traceback /c/zipvoice_data/exp/adapter_hbfix.log && { echo "[hbfix] training failed"; exit 1; }
sleep 30
echo "[hbfix] scoring 3000 $(date +%T)"
stage=2 stop_stage=3 system=adapter_hbfix3k exp_dir=$EXP num_iters=3000 project_low_band=0 \
  bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_adapter_hbfix3k.log 2>&1
echo "[hbfix] scored exit $? $(date +%T)"
