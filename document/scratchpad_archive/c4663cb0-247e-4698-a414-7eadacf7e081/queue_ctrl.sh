#!/bin/bash
# After stage 11 reaches 20000: score step 20000, then run the rollout-off
# control from checkpoint-10000 for 3000 steps and score its step 13000.
set -u
T="C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/tasks/b26wamygq.output"
EXP=/c/zipvoice_data/exp/adapter_steer
CTRL=/c/zipvoice_data/exp/adapter_steer_ctrl
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"

until grep -q "\[pause\] training exit" "$T"; do sleep 60; done
echo "[queue] training done: $(grep '\[pause\] training exit' "$T") $(date +%T)"
[ -f $EXP/checkpoint-20000.pt ] || { echo "[queue] no checkpoint-20000, stopping"; exit 1; }

avg_score () {  # exp_dir iter system
  python -m zipvoice.bin.generate_averaged_model --iter $2 --avg 2 \
    --model-name zipvoice_adapter --exp-dir $1 > /c/zipvoice_data/exp/avg_$3.log 2>&1 \
    || { echo "[queue] averaging $3 failed"; tail -3 /c/zipvoice_data/exp/avg_$3.log; return 1; }
  echo "[queue] scoring $3 $(date +%T)"
  stage=2 stop_stage=3 system=$3 exp_dir=$1 num_iters=$2 project_low_band=0 \
    bash run_adapter_libritts.sh > /c/zipvoice_data/exp/eval_$3.log 2>&1
  echo "[queue] scored $3 exit $? $(date +%T)"
}

avg_score $EXP 20000 adapter_steer20k

echo "[queue] control training $(date +%T)"
stage=11 stop_stage=11 steer_exp_dir=$CTRL steer_resume=$EXP/checkpoint-10000.pt \
  steer_rollout_prob=0 steer_num_iters=13000 \
  bash run_adapter_libritts.sh > /c/zipvoice_data/exp/adapter_steer_ctrl.log 2>&1
echo "[queue] control training exit $? $(date +%T)"
[ -f $CTRL/checkpoint-13000.pt ] || { echo "[queue] no control checkpoint-13000"; exit 1; }

avg_score $CTRL 13000 adapter_steerctrl13k
echo "[queue] done $(date +%T)"
