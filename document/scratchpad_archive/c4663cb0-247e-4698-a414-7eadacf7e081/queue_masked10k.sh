#!/bin/bash
# After Stage 12 is scored at 3000, score Stage 11's step-10000 weights with the
# write masked to the missing band: Stage 12 at step 0, no retraining.
set -u
OUT="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/tasks/b6fyyyx9t.output"
until grep -qE "scored 3000|averaging 3000 failed" "$OUT"; do sleep 60; done
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
echo "[masked] scoring $(date +%T)"
stage=2 stop_stage=3 system=adapter_steermasked10k exp_dir=/c/zipvoice_data/exp/adapter_steer_masked \
  num_iters=10000 project_low_band=0 bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_adapter_steermasked10k.log 2>&1
echo "[masked] scored exit $? $(date +%T)"
