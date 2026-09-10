#!/bin/bash
# Train stages 2 and 3, then synthesise and score. Stage 4 (the ablations) is
# skipped here: each is a second 30000-iteration run and they are only wanted
# once the main line has been shown to work.
set -e
cd "$(dirname "$0")"
stage=2 stop_stage=3 bash run_anyband_libritts.sh
stage=5 stop_stage=6 bash run_anyband_libritts.sh
