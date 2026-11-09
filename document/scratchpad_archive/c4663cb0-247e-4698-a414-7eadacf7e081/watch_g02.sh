#!/bin/bash
# For each checkpoint the gentle-GAN run saves, score sr8000 only and print one line.
export PYTHONUTF8=1
ROOT="G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice"
cd "$ROOT/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
exp=C:/zipvoice_data/exp/dual_rollout_g02
res=C:/zipvoice_data/exp/results
ev=C:/zipvoice_data/data/bwe_eval
em=C:/zipvoice_data/download/tts_eval_models
BF="$ROOT/experiments/adapter_multistep/egs/zipvoice/local/band_frames.py"
for it in 21000 22000 23000 24000 25000; do
  until [ -f $exp/checkpoint-${it}.pt ]; do sleep 30; done
  sleep 60
  out=$res/dualg02_${it}_sr8000
  python -m zipvoice.bin.infer_zipvoice --model-name zipvoice_dual --model-dir $exp/ \
    --checkpoint-name checkpoint-${it}.pt --tokenizer emilia \
    --test-list $ev/test_sr8000.tsv --res-dir $out \
    --vocoder-path C:/zipvoice_data/download/vocos-mel-24khz --num-step 16 > $out.log 2>&1
  sim=$(python -m zipvoice.eval.speaker_similarity.sim --wav-path $out --test-list $ev/test_score.tsv \
    --model-dir $em --score-path $out/sim_scores.tsv 2>&1 | grep -o "SIM-o score: [0-9.]*")
  mos=$(python -m zipvoice.eval.mos.utmos --wav-path $out --model-dir $em 2>&1 | grep -o "UTMOS score: [0-9.]*")
  bands=$(python "$BF" $out=x 2>&1 | grep "^x" | tr -s ' ')
  echo "g02 ${it} sr8000 | $sim | $mos | bands 4-6k/6-8k/8-12k speech silence swing: $bands"
done
echo WATCH_DONE
