#!/usr/bin/env bash
# The adapter arm's output configuration: a GAN-finetuned band adapter run with
# inference-time low-band projection ("projgan"). One model, one checkpoint, but
# --project-low-band 1 integrates two trajectories, so it costs two solves of
# that model: 2 x --num-step forwards. No second model and no second TTS pass.
#
# Scores (76 utterances, band_frames.py speech rows for the 8-12 kHz column):
#   sr8000  SIM-o 0.608  UTMOS 4.15  band -57.1  swing 40.8
#   sr16000 SIM-o 0.732  UTMOS 4.20  band -58.3  swing 35.2
# Average RTF 0.3762, against 0.185 for the same solve without projection.
#
# Note: --guidance-scale is left at the inference default of 1.0 here, not the
# 0.5 used by eval_ckpt.sh for the rollout checkpoints. Changing it invalidates
# the numbers above.
set -e

ROOT=${ZIPVOICE_DATA:-/c/zipvoice_data}
EXP=$ROOT/exp/adapter_gan
CKPT=iter-10000-avg-2.pt
SYS=projgan
RES=$ROOT/exp/results
EVAL=$ROOT/data/bwe_eval
EVAL_MODELS=$ROOT/download/tts_eval_models
SPK=$EVAL_MODELS/speaker_similarity/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin

export PYTHONPATH="$(pwd -W 2>/dev/null || pwd)"
export PYTHONUTF8=1

for rate in 8000 16000; do
  out=$RES/${SYS}_sr${rate}
  if [ ! -f "$out/.done" ]; then
    python -m zipvoice.bin.infer_zipvoice \
      --model-name zipvoice_adapter \
      --model-dir "$(cygpath -w $EXP)" \
      --checkpoint-name ${CKPT} \
      --tokenizer emilia \
      --test-list "$(cygpath -w $EVAL/test_sr${rate}.tsv)" \
      --res-dir "$(cygpath -w $out)" \
      --speaker-encoder-checkpoint "$(cygpath -w $SPK)" \
      --num-step 16 --t-shift 0.5 \
      --project-low-band 1 --prompt-floor 0 --residual-scale 1.0
    touch "$out/.done"
  fi
done

for rate in 8000 16000; do
  out=$RES/${SYS}_sr${rate}
  echo "=== ${SYS} @ sr${rate} ==="
  python -m zipvoice.eval.speaker_similarity.sim \
    --wav-path "$(cygpath -w $out)" \
    --test-list "$(cygpath -w $EVAL/test_score.tsv)" \
    --model-dir "$(cygpath -w $EVAL_MODELS)" \
    --score-path "$(cygpath -w $out/sim_scores.tsv)"
  python -m zipvoice.eval.mos.utmos \
    --wav-path "$(cygpath -w $out)" \
    --model-dir "$(cygpath -w $EVAL_MODELS)" \
    --score-path "$(cygpath -w $out/utmos_scores.tsv)"
done
