#!/bin/bash
# Evaluate the stage-2 (frozen-TTS) AnyBand checkpoint as its own arm, so that the
# stage-3 decoder finetune can be separated from the conditioning block itself.
# Queued behind the running RTF benchmark: synthesis timings and scoring must not
# contend for the same GPU.
set -e
cd "$(dirname "$0")"

while pgrep -f rtf_bench >/dev/null 2>&1; do sleep 30; done
sleep 20

anyband_arms="/c/zipvoice_data/exp/zipvoice_anyband_frozen:30000:anyband_frozen" \
  eval_systems="anyband_frozen" \
  stage=5 stop_stage=6 bash run_anyband_libritts.sh
