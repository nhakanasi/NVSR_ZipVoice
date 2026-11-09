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
# Floor the prompt's missing band at inference, as the mel-domain degradation
# does at training. A checkpoint trained without --resample-degrade has only
# seen that floor, never a real resampler's roll-off.
prompt_floor=${prompt_floor:-false}

# The degraded prompt conditions; 4000 is not scored. 24000 is the undegraded
# control: the branch is masked to zero there, so that arm is the released
# model and reproduces the stock numbers.
eval_rates=${eval_rates:-"8000 16000"}

# Solver steps at inference. 16 everywhere in the reported table; overridable
# because the branch is fitted on the true interpolation path, one step at a
# time, and then deployed inside a multi-step solve where its own residual has
# been entering ``x_t`` since the first step. Whether that compounding is what
# costs the identity is answerable by shortening the solve, provided the stock
# arm is resynthesised at the same step count -- fewer steps degrades every
# system, so only the gap between them carries the answer.
num_step=${num_step:-16}

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

# Where the branch starts. The released checkpoint, for an arm trained from
# scratch; a previous arm's averaged checkpoint, for one that only adds a term
# to an already-trained branch -- which is what isolates that term's effect from
# 30k steps of everything else.
init_checkpoint=${init_checkpoint:-${download_dir}/zipvoice/model.pt}
base_lr=${base_lr:-0.001}
# 0 samples cutoffs uniformly throughout, which is what a finetune off an
# already-trained branch wants: the easy-to-balanced ramp exists to get a fresh
# branch started.
curriculum_rho=${curriculum_rho:-0.7}

# Conditioning streams concatenated onto the frequency encoder's output: the
# prompt's time-pooled per-bin statistics (--use-speaker-cond) and the cutoff in
# Hz (--use-rate-cond). Off by default, which is the reported baseline; the
# ablations that added them are in ADAPTER.md.
use_speaker_cond=${use_speaker_cond:-0}
use_rate_cond=${use_rate_cond:-0}

# Whether the flow state handed to the network keeps the bins above the cutoff.
# It does by default, and that is a leak: ``x_t`` is built from the clean
# target, so the band the branch is meant to invent is already in the state it
# reads. Measured on the 30k checkpoint, removing those bins at inference time
# raises the endpoint error in the written band by 7.5x at t=0.1.
mask_xt=${mask_xt:-false}

# How often a checkpoint lands. 2000 was the 30k run's setting and it cost that
# whole run: stopping it at update 1350 left the experiment directory holding
# nothing but args.json. Each file is about 8 MB, so 1000 bounds the loss at
# roughly half an hour of compute for no meaningful disk cost.
save_every=${save_every:-1000}

# The arm name stages 2 and 3 write and score under. Every other arm is its own
# exp_dir, so those two stages are re-run over each with e.g.
# `system=adapter_lora exp_dir=${exp_root}/adapter_lora num_iters=4000`.
system=${system:-adapter}

# The speaker encoder handed to inference. A branch trained with the speaker
# stream and run without it is given a zero embedding flagged invalid, and the
# SIM-o that comes back understates the model. Harmless for arms trained without
# the stream, which ignore it.
speaker_ckpt=${speaker_ckpt:-${download_dir}/tts_eval_models/speaker_similarity/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin}

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
            --finetune 1 --base-lr ${base_lr} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every} --valid-interval ${save_every} \
            --checkpoint ${init_checkpoint} \
            --adapter-trainable adapter \
            --curriculum-steps ${num_iters} \
            --curriculum-rho ${curriculum_rho} \
            --full-band-prob 0.2 \
            --use-speaker-cond ${use_speaker_cond} \
            --use-rate-cond ${use_rate_cond} \
            --mask-xt ${mask_xt} \
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
                  --speaker-encoder-checkpoint ${speaker_ckpt} \
                  --num-step ${num_step} \
                  --prompt-floor ${prompt_floor}
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

# Stage 4: let the backbone learn to read what the branch writes.
#
# The frozen backbone loses identity the same way under every branch tried: the
# branch writes the missing band into the state, the frozen decoder reads it,
# and the decoder's own low band drifts. A branch outside the decoder can only
# answer that with more writes, and every such arm fell below stage 1.
#
# Here the decoder gets a rank-${lora_rank} LoRA update on its in/out
# projections, gated per utterance on whether the prompt is missing a band. A
# full-band prompt runs the released decoder bit for bit, so the 24 kHz output is
# stock by construction and the rollout's clean-prompt reference is still the
# frozen model's own solve while the LoRA trains.
#
# The decoder is also told what the branch wrote. The previous solver step's
# residual is projected to the decoder width and added after its input
# projection, so the LoRA adapts the backbone's output given the branch's
# output. The previous step's, not the current one's: the branch reads the
# decoder's hidden state, so the current residual exists only after the decoder
# has run, and feeding it back would take a second forward. This way it stays one
# decoder forward per step. On ordinary flow-matching batches there is no
# previous step and the input is empty, as at the first step of a solve.
#
# The branch and the LoRA train jointly from the released checkpoint: the
# branch's write masked to the missing bins as in stage 1, flow matching plus
# reconstruction on every ordinary batch, and from ${lora_rollout_start} on a
# fraction ${lora_rollout_prob} of batches are rollouts scored by L2 against the
# clean-prompt solve on both bands. ${lora_grad_steps} graphed steps keep the
# decoder in the graph, which is what sends the rollout's gradient into the LoRA
# factors and the residual projection.
lora_exp_dir=${lora_exp_dir:-${exp_root}/adapter_lora}
lora_rank=${lora_rank:-8}
lora_num_iters=${lora_num_iters:-20000}
lora_rollout_start=${lora_rollout_start:-2000}
lora_rollout_prob=${lora_rollout_prob:-0.25}
lora_grad_steps=${lora_grad_steps:-4}
lora_low_weight=${lora_low_weight:-1.0}
lora_high_weight=${lora_high_weight:-1.0}
# The band-level term: each missing bin's mean log-mel over the generated span
# against the recording's. Rollout L2 against the frozen solve lets the band go
# uniformly dark at a 4 kHz cutoff (-50 dB against -24 dB for real speech at
# 6-8 kHz by 14k steps), which this term charges directly. Underfill-only, so it
# pushes energy up and leaves a band that is too loud to the L2.
lora_band_level_weight=${lora_band_level_weight:-0}
lora_band_level_underfill=${lora_band_level_underfill:-false}
# Fixed rollout cutoffs, e.g. "4000 8000" for the sr8000 and sr16000 conditions,
# drawn with equal odds, and optionally one band-level weight per cutoff in the
# same order. Empty keeps the curriculum's cutoff on rollout batches.
lora_rollout_cutoffs=${lora_rollout_cutoffs:-}
lora_band_level_weights=${lora_band_level_weights:-}
# Degrade prompts by really resampling the waveform, as the evaluation prompts
# are, instead of flooring the mel above the cutoff; optionally with varied
# downsamplers for unknown deployment devices.
lora_resample_degrade=${lora_resample_degrade:-false}
lora_degrade_variants=${lora_degrade_variants:-false}
# Speaker identity on rollout batches, optionally scored on the prompt's band
# alone, to hold the low band's trajectory where the invented band disturbs it.
lora_identity_weight=${lora_identity_weight:-0}
lora_identity_low_band=${lora_identity_low_band:-false}
# A running cost along the rollout's solve (VGG-Flow): the adapted velocity
# against the stock one at the same state and degraded prompt, on the observed
# bins. It holds the low band's trajectory at every step rather than scoring the
# endpoint, where the clean-prompt reference puts a floor under the low L2.
lora_velocity_low_weight=${lora_velocity_low_weight:-0}
lora_velocity_low_steps=${lora_velocity_low_steps:-16}
# A checkpoint to resume an interrupted run from; empty starts fresh.
lora_resume=${lora_resume:-}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Train the branch and a band-gated decoder LoRA jointly"
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr ${base_lr} \
            --num-iters ${lora_num_iters} \
            --save-every-n ${save_every} --valid-interval ${save_every} \
            --checkpoint ${init_checkpoint} \
            --adapter-trainable lora --lora-rank ${lora_rank} \
            --curriculum-rho ${curriculum_rho} \
            --full-band-prob 0.2 \
            --use-speaker-cond ${use_speaker_cond} \
            --use-rate-cond ${use_rate_cond} \
            --mask-xt ${mask_xt} \
            --steer-all-bins 0 \
            --identity-rollout-steps ${num_step} \
            --identity-rollout-prob ${lora_rollout_prob} \
            --identity-rollout-start-step ${lora_rollout_start} \
            --rollout-distance l2 \
            --identity-rollout-grad-steps ${lora_grad_steps} \
            --rollout-low-weight ${lora_low_weight} \
            --rollout-high-weight ${lora_high_weight} \
            --rollout-band-level-weight ${lora_band_level_weight} \
            --rollout-band-level-underfill ${lora_band_level_underfill} \
            ${lora_rollout_cutoffs:+--rollout-cutoffs ${lora_rollout_cutoffs// /,}} \
            ${lora_band_level_weights:+--rollout-band-level-weights ${lora_band_level_weights// /,}} \
            --resample-degrade ${lora_resample_degrade} \
            --degrade-variants ${lora_degrade_variants} \
            --identity-rollout-weight ${lora_identity_weight} \
            --identity-rollout-low-band ${lora_identity_low_band} \
            --rollout-velocity-low-weight ${lora_velocity_low_weight} \
            --rollout-velocity-low-steps ${lora_velocity_low_steps} \
            --speaker-encoder-checkpoint ${speaker_ckpt} \
            ${lora_resume:+--resume-from ${lora_resume}} \
            --exp-dir ${lora_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${lora_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${lora_exp_dir}
fi

# Score it:
#   stage=2 stop_stage=3 system=adapter_lora exp_dir=${exp_root}/adapter_lora \
#         num_iters=4000 bash run_adapter_libritts.sh
