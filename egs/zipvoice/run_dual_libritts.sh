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

# Weights that would otherwise be fetched from HuggingFace. The offline bundle
# sets ZIPVOICE_ASSETS to the folder it ships them in; without it they are read
# from the download directory the preparation stages write.
assets_dir=${ZIPVOICE_ASSETS:-${download_dir}}
backbone_dir=${assets_dir}/zipvoice
# Vocos, passed by path so inference never calls Vocos.from_pretrained. Empty
# means "let inference fetch it", which only works with a network.
vocoder_path=${vocoder_path:-${assets_dir}/vocos-mel-24khz}

tokenizer=emilia
token_file=${backbone_dir}/tokens.txt

exp_dir=${exp_dir:-${exp_root}/dual}
system=${system:-dual}
# conf/zipvoice_dual_velocity.json has the branch predict a velocity added to
# its own state, x1 = y + (1 - t) * v, instead of the endpoint directly. The
# endpoint form has no identity path from the state and never reproduced the
# reference's spectral detail, even from a 95%-clean state. Same branch size.
model_config=${model_config:-conf/zipvoice_dual.json}
num_iters=${num_iters:-30000}
save_every=${save_every:-1000}
base_lr=${base_lr:-0.001}
# Fine-tuning holds base_lr fixed. From this step it decays (cosine) to
# lr_final_ratio * base_lr by num_iters, so the last checkpoints settle rather
# than being snapshots of a run still moving. Negative keeps it fixed.
lr_decay_start=${lr_decay_start:--1}
lr_final_ratio=${lr_final_ratio:-0.1}
# Rollout adds no-gradient backbone/branch forwards but the supervised branch
# activations still dominate peak VRAM. 48 s keeps the measured peak inside the
# 10-12 GB target on a 16 GB card; override after measuring on another GPU.
max_duration=${max_duration:-48}

# The released checkpoint. Only branch.* is trained; every released tensor is
# excluded from the optimizer, not merely zero-gradient, because ScaledAdam
# would otherwise shrink it through weight decay.
init_checkpoint=${init_checkpoint:-${backbone_dir}/model.pt}

# The reconstruction term. Small: flow matching carries the arm, and an L1 on the
# endpoint is exactly the term whose optimum is the conditional median, which in
# log-mel sits 9-10 dB under any real sample. It is here to keep the branch's
# endpoint anchored early, not to be the objective.
recon_loss_weight=${recon_loss_weight:-0.1}

# Where the branch's MSE is measured. "endpoint" is the velocity error weighted
# by (1-t)^2: the late steps, which have to remove the last of the noise, barely
# train. "velocity" weights it uniformly in t, as the backbone was trained, and
# needs t_max below 1 so the 1/(1-t)^2 stays bounded.
fm_space=${fm_space:-endpoint}
t_max=${t_max:-1.0}

# Scheduled exposure to the state distribution used by inference. Selected
# examples start at pure noise, preventing target high-band content from
# entering through either the branch state or the frozen backbone memory. The
# memory is computed on the branch's own state, as at inference. Off for the
# first steps, where a fresh branch's rollouts are noise, then ramped in.
rollout_prob=${rollout_prob:-0.5}
rollout_steps=${rollout_steps:-4}
rollout_start_step=${rollout_start_step:-3000}
rollout_ramp_steps=${rollout_ramp_steps:-3000}

# Degrade prompts by really resampling the waveform, as the evaluation prompts
# are. The mel floor is not what a 24k->8k->24k file looks like above the cutoff.
resample_degrade=${resample_degrade:-true}

# Band-level term: per sub-band of the missing range, match the target's mean
# energy over the generated frames (level and tilt) and its spread over them
# (how far the band follows the voice). Off by default; it is the alternative
# to the discriminators for getting the invented band's level right.
band_loss_weight=${band_loss_weight:-0.0}

# The same statistics plus a smoothed per-band envelope, taken on the mel the
# evaluation solve produces (16 steps, t_shift 0.5, guidance 1.0) for
# solve_utts utterances per batch, instead of on a one-shot endpoint estimate.
solve_loss_weight=${solve_loss_weight:-0.0}
solve_utts=${solve_utts:-2}

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
# "solve" judges the evaluation-matched solve of the solve_utts utterances,
# over the target's own low band, instead of a one-shot endpoint estimate. It
# reuses the solve solve_loss_weight pays for; keep that weight above zero as
# the level anchor.
gan_source=${gan_source:-endpoint}

# Continue a run from one of its mid-epoch checkpoints (for example the last one
# before the GAN starts), restoring the optimizer and the step count.
resume_checkpoint=${resume_checkpoint:-}

# Cutoff sampling: easy (nearly full band) to balanced over the run, so a fresh
# branch is not asked to invent 10 kHz of spectrum on its first step.
cutoff_curriculum_steps=${cutoff_curriculum_steps:-${num_iters}}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the high-band branch against the frozen backbone"
      ${PY} -m zipvoice.bin.train_zipvoice_dual \
            --world-size 1 --use-fp16 1 --max-duration ${max_duration} --max-len 20 \
            --model-config ${model_config} \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --finetune 1 --base-lr ${base_lr} \
            --lr-decay-start ${lr_decay_start} --lr-final-ratio ${lr_final_ratio} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every} \
            --checkpoint ${init_checkpoint} \
            --recon-loss-weight ${recon_loss_weight} \
            --fm-space ${fm_space} --t-max ${t_max} \
            --rollout-prob ${rollout_prob} \
            --rollout-steps ${rollout_steps} \
            --rollout-start-step ${rollout_start_step} \
            --rollout-ramp-steps ${rollout_ramp_steps} \
            --resample-degrade ${resample_degrade} \
            --band-loss-weight ${band_loss_weight} \
            --solve-loss-weight ${solve_loss_weight} --solve-utts ${solve_utts} \
            --cutoff-curriculum-steps ${cutoff_curriculum_steps} \
            --gan-start-step ${gan_start_step} \
            --adv-loss-weight ${adv_loss_weight} \
            --feat-match-loss-weight ${feat_match_loss_weight} \
            --disc-lr ${disc_lr} \
            --gan-t-min ${gan_t_min} --gan-source ${gan_source} \
            ${resume_checkpoint:+--resume-checkpoint ${resume_checkpoint}} \
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
                  --vocoder-path ${vocoder_path} \
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
