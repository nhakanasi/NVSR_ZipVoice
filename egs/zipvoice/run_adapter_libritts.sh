#!/bin/bash

# Train and evaluate the band adapter: a small side branch on a frozen released
# ZipVoice, which writes a velocity residual into the mel bins a band-limited
# prompt does not observe and is multiplied by the band mask, so at full band it
# contributes exactly zero and the model is the released checkpoint.
#
# Data preparation and the band-limited evaluation set are shared with the
# extender study and are not repeated here. Run stages 1-4 and stage 8 of
# run_bwe_libritts.sh first; this script starts from the features and the
# evaluation prompts they produce, and writes into the same result root so the
# existing scoring stage sees the new arm.

# Emilia tokens are IPA. lhotse writes manifests with the locale codec, which on
# a Windows machine is cp1252 and cannot encode them, so force UTF-8 mode.
export PYTHONUTF8=1

set -e
set -u
set -o pipefail

# Windows ships a python3.exe App Execution Alias that is on PATH but only
# prints an install prompt, so probe by running it, not by command -v.
if python3 -c "" >/dev/null 2>&1; then PY=python3; else PY=python; fi

if ${PY} -c "import sys; sys.exit(0 if sys.platform == 'win32' else 1)"; then
      export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
else
      export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"
fi

stage=${stage:-1}
stop_stage=${stop_stage:-3}

# The degraded prompt conditions. 24000 is the undegraded control: the branch is
# masked to zero there, so that arm is the released model and is expected to
# reproduce the stock numbers exactly.
eval_rates=${eval_rates:-"4000 8000 16000"}

wandb=${wandb:-0}
wandb_project=${wandb_project:-zipvoice-adapter}

data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
download_dir=${data_root}/download
fbank_dir=${data_root}/data/fbank
exp_root=${data_root}/exp
eval_dir=${data_root}/data/bwe_eval
results_dir=${exp_root}/results

tokenizer=emilia
token_file=${download_dir}/zipvoice/tokens.txt

exp_dir=${exp_dir:-${exp_root}/adapter}
num_iters=${num_iters:-30000}

# Extra conditioning streams, concatenated onto the frequency encoder's output.
# They are part of what the branch is *given*, not of what it is scored on, so
# they belong in the branch's own training rather than in a finetune: a branch
# trained for 30k steps without F0 has already learnt to do without it.
#
#   --use-f0-cond    the prompt's pitch track, estimated from the band-limited
#                    condition mel. Not from the waveform: training band-limits
#                    only in the mel domain, so the audio in the batch is clean
#                    and full band, and a tracker run on it would be reading the
#                    high band the branch exists to invent.
#   --use-rate-cond  the cutoff in Hz, sinusoidally embedded, broadcast over
#                    time. How far the extrapolation has to reach, as a number.
#
# The ablation is one full run per arm, each with its own exp_dir and system:
#   exp_dir=${exp_root}/adapter      system=adapter      f0=0 rate=0  (baseline)
#   exp_dir=${exp_root}/adapter_f0   system=adapter_f0   f0=1 rate=0
#   exp_dir=${exp_root}/adapter_rate system=adapter_rate f0=0 rate=1
#   exp_dir=${exp_root}/adapter_cond system=adapter_cond f0=1 rate=1
use_f0_cond=${use_f0_cond:-1}
use_rate_cond=${use_rate_cond:-1}

# The arm name stages 2 and 3 write and score under. Every other arm -- a
# conditioning-stream ablation, or stage 4's finetune -- is its own exp_dir, so
# those two stages are re-run over each with e.g.
# `system=adapter_id exp_dir=${exp_root}/adapter_id num_iters=10000`.
system=${system:-adapter}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the adapter against the frozen released model"
      # --adapter-trainable adapter leaves every released tensor out of the
      # optimizer, not merely zero-gradient: ScaledAdam keeps a per-tensor scale,
      # so exclusion is what guarantees the released weights are untouched.
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.001 \
            --num-iters ${num_iters} \
            --save-every-n 2000 --valid-interval 2000 \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --adapter-trainable adapter \
            --curriculum-steps ${num_iters} \
            --full-band-prob 0.2 \
            --use-f0-cond ${use_f0_cond} \
            --use-rate-cond ${use_rate_cond} \
            --exp-dir ${exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Synthesise from band-limited prompts"
      # model.json and tokens.txt are written into the experiment directory by
      # the trainer. model.json in particular is the *resolved* config, carrying
      # whichever conditioning streams that run enabled, so copying the template
      # over it here would build an inference model missing them and the
      # checkpoint would not load.
      for rate in ${eval_rates}; do
            ${PY} -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice_adapter \
                  --model-dir ${exp_dir}/ \
                  --checkpoint-name iter-${num_iters}-avg-2.pt \
                  --tokenizer ${tokenizer} \
                  --test-list ${eval_dir}/test_sr${rate}.tsv \
                  --res-dir ${results_dir}/${system}_sr${rate} \
                  --num-step 16
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Score the adapter arm"
      # Scoring lives in run_bwe_libritts.sh; only the arm list differs, so that
      # stage is reused rather than duplicated. The stock and cascade arms it
      # compares against were scored by earlier runs against the same evaluation
      # set and are not resynthesised.
      eval_systems="${eval_systems:-${system}}" \
            stage=10 stop_stage=10 eval_rates="${eval_rates}" \
            bash run_bwe_libritts.sh
fi

# Stage 4 finetunes the trained branch against two further losses, measured on
# audio rather than on the mel. The 30k run above raised UTMOS at a 4 kHz cutoff
# from 2.80 to 3.96 but dropped SIM-o below both baselines at every rate, by more
# than ten times the n=76 noise floor at 4 kHz. Neither of its loss terms carries
# speaker identity: flow matching scores the frozen backbone's velocity, and the
# L1 is minimised by whatever high band is closest on average, which belongs to
# nobody. So the branch is charged for the speaker distance between the vocoded
# endpoint estimate and the vocoded target.
#
# Weights are calibrated, not guessed. Run once with them at 1e-6, which builds
# the losses and logs their magnitudes while leaving the objective effectively
# untouched, then set each so its contribution is roughly 10-30% of the total.
id_exp_dir=${id_exp_dir:-${exp_root}/adapter_id}
id_num_iters=${id_num_iters:-10000}
identity_weight=${identity_weight:-1e-6}
fmatch_weight=${fmatch_weight:-1e-6}
speaker_ckpt=${speaker_ckpt:-${download_dir}/tts_eval_models/speaker_similarity/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Finetune with identity and feature matching losses"
      # --curriculum-rho 0 samples cutoffs uniformly from the start: the
      # curriculum exists to ease a fresh branch into wide bandwidths, and this
      # branch is already trained. Half the learning rate of stage 1, since this
      # is meant to correct the residual, not relearn it.
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.0005 \
            --num-iters ${id_num_iters} \
            --save-every-n 2000 --valid-interval 2000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --identity-loss-weight ${identity_weight} \
            --fmatch-loss-weight ${fmatch_weight} \
            --speaker-encoder-checkpoint ${speaker_ckpt} \
            --exp-dir ${id_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${id_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${id_exp_dir}
fi
