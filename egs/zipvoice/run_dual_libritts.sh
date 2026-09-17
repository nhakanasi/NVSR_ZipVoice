#!/bin/bash

# Train and evaluate the dual-diffusion arm: a frozen released ZipVoice plus a
# second flow-matching branch that solves its own trajectory for the mel bins a
# band-limited prompt does not observe. The backbone's trajectory is never
# written to; at every solver step the branch's observed band is projected back
# onto it, so the low band -- and with it the speaker identity -- is the released
# model's by construction rather than by a trained constraint.
#
# Data preparation and the band-limited evaluation set are shared with the
# adapter study and are not repeated here: run stages 1-4 and stage 8 of
# run_bwe_libritts.sh first. This script starts from the features and the
# evaluation prompts they produce and writes into the same result root.

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

# Prompt sampling rates, i.e. cutoffs of 2, 4 and 8 kHz. The 8 kHz one is what
# this design exists to fix: the adapter arm reached only +4.6 dB there against a
# +10 dB bar.
eval_rates=${eval_rates:-"4000 8000 16000"}
num_step=${num_step:-16}

data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
download_dir=${data_root}/download
fbank_dir=${data_root}/data/fbank
exp_root=${data_root}/exp
eval_dir=${data_root}/data/bwe_eval
results_dir=${exp_root}/results
eval_model_dir=${download_dir}/tts_eval_models

tokenizer=emilia
token_file=${download_dir}/zipvoice/tokens.txt

exp_dir=${exp_dir:-${exp_root}/dual}
system=${system:-dual}
num_iters=${num_iters:-30000}
save_every=${save_every:-1000}
base_lr=${base_lr:-0.001}

# The released checkpoint. Only branch.* is trained; every released tensor is
# excluded from the optimizer, not merely zero-gradient, because ScaledAdam
# would otherwise shrink it through weight decay.
init_checkpoint=${init_checkpoint:-${download_dir}/zipvoice/model.pt}

# The reconstruction term. Small: flow matching carries the arm, and an L1 on the
# endpoint is exactly the term whose optimum is the conditional median, which in
# log-mel sits 9-10 dB under any real sample. It is here to keep the branch's
# endpoint anchored early, not to be the objective.
recon_loss_weight=${recon_loss_weight:-0.1}

# Stage 2 of the objective: the cross-band and spectrogram discriminators, which
# ask whether *this* invented band follows *this* observed low band. Off until
# the branch can already produce a plausible band on its own, and applied only to
# the later half of the flow-time grid, where the endpoint estimate is worth
# judging. Learning rate 6e-5 rather than the vocoder-GAN 2e-4: at 2e-4 the
# discriminator separates within 2k steps and the generator term, which has no
# lower bound, runs away.
gan_start_step=${gan_start_step:-20000}
adv_loss_weight=${adv_loss_weight:-0.1}
feat_match_loss_weight=${feat_match_loss_weight:-2.0}
disc_lr=${disc_lr:-6e-5}
gan_t_min=${gan_t_min:-0.5}

# Cutoff sampling: easy (nearly full band) to balanced over the run, so a fresh
# branch is not asked to invent 10 kHz of spectrum on its first step.
cutoff_curriculum_steps=${cutoff_curriculum_steps:-${num_iters}}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the high-band branch against the frozen backbone"
      ${PY} -m zipvoice.bin.train_zipvoice_dual \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_dual.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --finetune 1 --base-lr ${base_lr} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every} \
            --checkpoint ${init_checkpoint} \
            --recon-loss-weight ${recon_loss_weight} \
            --cutoff-curriculum-steps ${cutoff_curriculum_steps} \
            --gan-start-step ${gan_start_step} \
            --adv-loss-weight ${adv_loss_weight} \
            --feat-match-loss-weight ${feat_match_loss_weight} \
            --disc-lr ${disc_lr} \
            --gan-t-min ${gan_t_min} \
            --exp-dir ${exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} --avg 2 \
            --model-name zipvoice_dual \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Synthesise from band-limited prompts"
      # model.json and tokens.txt are written into the experiment directory by
      # the trainer, and model.json is the *resolved* config carrying the dual
      # block this run used, so the template must not be copied over it.
      #
      # The cutoff is not passed here: inference reads the prompt file's own
      # sampling rate, since resampling to 24 kHz does not recreate a band the
      # recording never had.
      for rate in ${eval_rates}; do
            ${PY} -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice_dual \
                  --model-dir ${exp_dir}/ \
                  --checkpoint-name iter-${num_iters}-avg-2.pt \
                  --tokenizer ${tokenizer} \
                  --test-list ${eval_dir}/test_sr${rate}.tsv \
                  --res-dir ${results_dir}/${system}_sr${rate} \
                  --num-step ${num_step}
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Score the dual arm"
      # Scored against test_score.tsv, which points at the original 24 kHz
      # prompt: scoring against the degraded one would measure the degradation
      # instead of the synthesis.
      for rate in ${eval_rates}; do
            wav_path=${results_dir}/${system}_sr${rate}
            [ -d "${wav_path}" ] || continue
            echo "--- ${system} @ ${rate} Hz ---"

            ${PY} -m zipvoice.eval.speaker_similarity.sim \
                  --wav-path ${wav_path} \
                  --test-list ${eval_dir}/test_score.tsv \
                  --model-dir ${eval_model_dir} \
                  --score-path ${wav_path}/sim_scores.tsv

            ${PY} -m zipvoice.eval.mos.utmos \
                  --wav-path ${wav_path} \
                  --model-dir ${eval_model_dir}

            ${PY} -m zipvoice.eval.wer.seedtts \
                  --wav-path ${wav_path} \
                  --test-list ${eval_dir}/test_score.tsv \
                  --decode-path ${wav_path}/decode.txt \
                  --model-dir ${eval_model_dir} \
                  --lang en
      done

      # The third leg of the gate. Each row is that system's own band energies
      # relative to its 100-2000 Hz band, so the 8-12k column against the stock
      # arm's is the +10 dB number.
      echo "--- band energies ---"
      profile_args=""
      for rate in ${eval_rates}; do
            for arm in stock ${system}; do
                  d=${results_dir}/${arm}_sr${rate}
                  [ -d "${d}" ] && profile_args="${profile_args} ${d}=${arm}_sr${rate}"
            done
      done
      # shellcheck disable=SC2086
      [ -n "${profile_args}" ] && ${PY} local/band_profile.py ${profile_args}
fi
