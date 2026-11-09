#!/bin/bash
# Mel comparison for the median and worst SIM change of $1 against steer10k,
# with a row per scored steer checkpoint. Usage: mels.sh 16k
set -u
NEW=$1
R=C:/zipvoice_data/exp/results
D=C:/zipvoice_data/data/bwe_eval
cd "G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice/experiments/adapter_multistep/egs/zipvoice"
OUT=mel_compare
for rate in 4000 8000 16000; do
  picks=$(python -c "
import sys
R, rate, new = sys.argv[1:4]
load = lambda s: dict(l.strip().split(chr(9)) for l in open(f'{R}/adapter_steer{s}_sr{rate}/sim_scores.tsv').read().splitlines()[1:] if chr(9) in l)
a, b = load('10k'), load(new)
d = sorted((float(b[k]) - float(a[k]), k) for k in a if k in b)
sys.stdout.write('median:' + d[len(d) // 2][1] + ' worst:' + d[0][1])
" $R $rate $NEW)
  for tag in $picks; do
    kind=${tag%%:*}; n=${tag#*:}
    items=(--item "prompt=$D/prompts/sr$rate/$n.wav" --item "ground truth=$D/references/$n.wav")
    for k in 10k 13k 16k 20k ctrl13k; do
      [ -f $R/adapter_steer${k}_sr$rate/$n.wav ] && items+=(--item "steer$k=$R/adapter_steer${k}_sr$rate/$n.wav")
    done
    python local/plot_bwe_mel.py --output $OUT/steer${NEW}_sr${rate}_${kind}_sim_change.png \
      --title "sr$rate, $kind SIM change 10k to $NEW: $n" --cutoff-khz $(python -c "print($rate/2000)") "${items[@]}" \
      && echo "wrote $OUT/steer${NEW}_sr${rate}_${kind}_sim_change.png"
  done
done
