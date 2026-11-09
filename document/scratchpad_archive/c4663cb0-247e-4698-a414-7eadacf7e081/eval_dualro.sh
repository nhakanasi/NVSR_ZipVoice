#!/bin/bash
# Score one raw checkpoint of the dual rollout run, as run_dual_libritts.sh stages 2-3 do.
set -eu
export PYTHONUTF8=1
ROOT="G:/Other computers/My Laptop/OneDrive - Hanoi University of Science and Technology/Viettel/GD2/ZipVoice"
cd "$ROOT/egs/zipvoice"
export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
it=$1
exp=/c/zipvoice_data/exp/dual_rollout
res=/c/zipvoice_data/exp/results
ev=/c/zipvoice_data/data/bwe_eval
em=/c/zipvoice_data/download/tts_eval_models
until [ -f $exp/checkpoint-${it}.pt ]; do sleep 20; done
sleep 30  # let the save finish
for rate in 8000 16000; do
  out=$res/dualro${it}_sr${rate}
  python -m zipvoice.bin.infer_zipvoice --model-name zipvoice_dual --model-dir $exp/ \
    --checkpoint-name checkpoint-${it}.pt --tokenizer emilia \
    --test-list $ev/test_sr${rate}.tsv --res-dir $out \
    --vocoder-path /c/zipvoice_data/dist/zipvoice_dual_offline/assets/vocos-mel-24khz --num-step 16
  echo "=== dualro${it} sr${rate} ==="
  python -m zipvoice.eval.speaker_similarity.sim --wav-path $out --test-list $ev/test_score.tsv \
    --model-dir $em --score-path $out/sim_scores.tsv 2>&1 | grep -i "sim"
  python -m zipvoice.eval.mos.utmos --wav-path $out --model-dir $em 2>&1 | grep -i "utmos"
done
BF="$ROOT/experiments/adapter_multistep/egs/zipvoice/local/band_frames.py"
python "$BF" C:/zipvoice_data/data/bwe_eval/references=ref C:/zipvoice_data/exp/results/dualro${it}_sr8000=sr8000 C:/zipvoice_data/exp/results/dualro${it}_sr16000=sr16000
P="$ROOT/experiments/adapter_multistep/egs/zipvoice/local/plot_bwe_mel.py"
u=1272_1272_135031_000008_000000-5649
for rate in 8000 16000; do
  python "$P" --output "$ROOT/egs/zipvoice/mel_dualro${it}_sr${rate}.png" --cutoff-khz $((rate/2000))     --title "dual rollout ckpt ${it}, sr${rate} prompt ($u)"     --item "reference=C:/zipvoice_data/data/bwe_eval/references/$u.wav"     --item "prompt sr${rate}=C:/zipvoice_data/data/bwe_eval/prompts/sr${rate}/$u.wav"     --item "dualro 6000=C:/zipvoice_data/exp/results/dualro6000_sr${rate}/$u.wav"     --item "dualro ${it}=C:/zipvoice_data/exp/results/dualro${it}_sr${rate}/$u.wav"
done
echo EVAL_DONE
