#!/bin/bash
# Score Stage 12 at 2000 (average over 1000-2000; --avg 2 needs a third
# checkpoint that does not exist yet), resume to 3000, score that with --avg 2.
set -u
EXP=/c/zipvoice_data/exp/adapter_hbfix
LOG=/c/zipvoice_data/exp/adapter_hbfix.log
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
score() {  # iter avg
  python -m zipvoice.bin.generate_averaged_model --iter $1 --avg $2 \
    --model-name zipvoice_adapter --exp-dir $EXP > /c/zipvoice_data/exp/hbfix_avg_$1.log 2>&1 \
    || { echo "[hbfix] averaging $1 failed"; tail -3 /c/zipvoice_data/exp/hbfix_avg_$1.log; return; }
  # The scoring stage reads iter-N-avg-2.pt by name.
  [ $2 = 2 ] || cp $EXP/iter-$1-avg-$2.pt $EXP/iter-$1-avg-2.pt
  echo "[hbfix] scoring $1 avg $2 $(date +%T)"
  stage=2 stop_stage=3 system=adapter_hbfix${1%000}k exp_dir=$EXP num_iters=$1 project_low_band=0 \
    bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_adapter_hbfix${1%000}k.log 2>&1
  echo "[hbfix] scored $1 exit $? $(date +%T)"
}
score 2000 1
echo "[hbfix] resuming $(date +%T)"
stage=12 stop_stage=12 hbfix_resume=$EXP/checkpoint-2000.pt bash run_adapter_libritts.sh >> $LOG 2>&1
echo "[hbfix] training exit $? $(date +%T)"
score 3000 2
