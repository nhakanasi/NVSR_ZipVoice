#!/bin/bash
# Stop Stage 12 once checkpoint-2000 is on disk, score it as adapter_hbfix2k,
# then resume to 3000 and score that as adapter_hbfix3k.
set -u
EXP=/c/zipvoice_data/exp/adapter_hbfix
LOG=/c/zipvoice_data/exp/adapter_hbfix.log
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
until [ -f $EXP/checkpoint-2000.pt ]; do sleep 20; done
# The checkpoint is written in one torch.save; wait for its size to settle.
prev=0; while s=$(stat -c %s $EXP/checkpoint-2000.pt); [ "$s" != "$prev" ]; do prev=$s; sleep 15; done
for pid in $(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train_zipvoice_adapter' -and \$_.CommandLine -match 'adapter_hbfix' } | ForEach-Object { \$_.ProcessId }" | tr -d ''); do taskkill //T //F //PID $pid; done
pkill -f "stage=12" 2>/dev/null
sleep 10
echo "[hbfix] stopped $(date +%T)"
score() {
  python -m zipvoice.bin.generate_averaged_model --iter $1 --avg 2 \
    --model-name zipvoice_adapter --exp-dir $EXP > /c/zipvoice_data/exp/hbfix_avg_$1.log 2>&1 \
    || { echo "[hbfix] averaging $1 failed"; tail -3 /c/zipvoice_data/exp/hbfix_avg_$1.log; return; }
  echo "[hbfix] scoring $1 $(date +%T)"
  stage=2 stop_stage=3 system=adapter_hbfix${1%000}k exp_dir=$EXP num_iters=$1 project_low_band=0 \
    bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_adapter_hbfix${1%000}k.log 2>&1
  echo "[hbfix] scored $1 exit $? $(date +%T)"
}
score 2000
echo "[hbfix] resuming $(date +%T)"
stage=12 stop_stage=12 hbfix_resume=$EXP/checkpoint-2000.pt bash run_adapter_libritts.sh >> $LOG 2>&1
echo "[hbfix] training exit $? $(date +%T)"
score 3000
