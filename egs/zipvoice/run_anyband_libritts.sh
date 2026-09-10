#!/bin/bash

# Fine-tune the released ZipVoice checkpoint into ZipVoiceAnyBand on LibriTTS.
#
# The model conditions the later ODE steps on an AnyBand-style description of
# the band-limited prompt -- a frequency-axis encoding of which mel bins are
# observed, plus an F0 contour and a voicing flag -- and lets the flow matching
# fill the missing band in, rather than reconstructing that band with a separate
# module as the bandwidth extender does.
#
# Data preparation and the band-limited evaluation set are shared with the
# extender study and are not repeated here. Run stages 1-4 and stage 8 of
# run_bwe_libritts.sh first; this script starts from the features and the
# evaluation prompts they produce, and writes into the same result root so the
# existing scoring stage sees the new arms.

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
stop_stage=${stop_stage:-6}

# The reported table is 8 / 16 / 22.05 kHz. 24000 is the undegraded control and
# is available by overriding this, but it is not part of the headline result.
eval_rates=${eval_rates:-"8000 16000 22050"}

# Set wandb=1 and export WANDB_API_KEY (or run `wandb login`) to mirror the
# tensorboard metrics to Weights & Biases. Never put the key in this file.
wandb=${wandb:-0}
wandb_project=${wandb_project:-zipvoice-anyband}

data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
download_dir=${data_root}/download
fbank_dir=${data_root}/data/fbank
exp_root=${data_root}/exp
eval_dir=${data_root}/data/bwe_eval
results_dir=${exp_root}/results

tokenizer=emilia
token_file=${download_dir}/zipvoice/tokens.txt

world_size=1
max_duration=72
max_len=20

exp_dir=${exp_root}/zipvoice_anyband
f0_dir=${exp_root}/f0_estimator

# The manifest the F0 estimator trains on. It has to be the pre-tokenisation
# cut set, because that one still carries recordings and pYIN needs a waveform.
f0_manifest=${f0_manifest:-${fbank_dir}/libritts_cuts_train-clean-360.jsonl.gz}
f0_cuts=${f0_cuts:-4000}
f0_epochs=${f0_epochs:-20}

# Stage 2 trains only the new modules, so every released tensor stays exactly as
# it was; stage 3 then lets the flow-matching decoder adapt at a low learning
# rate. Splitting it this way is what keeps the pretrained model intact: the
# conditioning path is already useful before anything released is allowed to
# move.
frozen_iters=${frozen_iters:-30000}
joint_iters=${joint_iters:-15000}

save_every_n=${save_every_n:-1000}
valid_interval=${valid_interval:-2000}

resume_from=${resume_from:-}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the mel-domain F0 and voicing estimator"
      # Trained once against pYIN targets and frozen thereafter. It is a fixed
      # feature extractor inside the flow-matching model, which is why it can be
      # trained on a few thousand utterances rather than the whole corpus.
      ${PY} -m zipvoice.bin.train_f0_estimator \
            --manifest ${f0_manifest} \
            --num-cuts ${f0_cuts} \
            --num-epochs ${f0_epochs} \
            --exp-dir ${f0_dir}
fi

common_args=(
      --world-size ${world_size}
      --use-fp16 1
      --max-duration ${max_duration}
      --max-len ${max_len}
      --model-config conf/zipvoice_anyband.json
      --tokenizer ${tokenizer}
      --token-file ${token_file}
      --dataset libritts
      --manifest-dir ${fbank_dir}
      --wandb ${wandb}
      --wandb-project ${wandb_project}
      --f0-estimator-checkpoint ${f0_dir}/best.pt
)

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train the AnyBand conditioning against the frozen TTS"
      # --anyband-trainable anyband leaves every released tensor untouched, so
      # this stage cannot damage the pretrained model even in principle: the
      # frozen tensors are dropped from the optimizer entirely and the
      # per-tensor scale of ScaledAdam never reaches them.
      ${PY} -m zipvoice.bin.train_zipvoice_anyband \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.001 \
            --num-iters ${frozen_iters} \
            --save-every-n ${save_every_n} \
            --valid-interval ${valid_interval} \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --anyband-trainable anyband \
            --curriculum-steps ${frozen_iters} \
            --exp-dir ${exp_dir}_frozen

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${frozen_iters} --avg 2 \
            --model-name zipvoice_anyband \
            --exp-dir ${exp_dir}_frozen
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Light joint fine-tuning of the flow-matching decoder"
      # The text encoder and the token embedding stay frozen: they see no band
      # information, and the released model was tuned around that text
      # representation. Cutoffs are sampled uniformly here (curriculum off), so
      # the decoder adapts to the whole range rather than to the easy end of it.
      if [ -n "${resume_from}" ]; then
            start_args=(--resume-from "${resume_from}")
      else
            start_args=(--checkpoint ${exp_dir}_frozen/iter-${frozen_iters}-avg-2.pt)
      fi
      ${PY} -m zipvoice.bin.train_zipvoice_anyband \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.0005 \
            --num-iters ${joint_iters} \
            --save-every-n ${save_every_n} \
            --valid-interval ${valid_interval} \
            "${start_args[@]}" \
            --anyband-trainable decoder \
            --curriculum-rho 0.0 \
            --curriculum-steps 0 \
            --exp-dir ${exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${joint_iters} --avg 2 \
            --model-name zipvoice_anyband \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Ablations, each a stage 2 rerun with one part removed"
      # Every ablation stops where stage 2 stops rather than continuing through
      # stage 3. That holds the released weights fixed across all of them, so a
      # difference is attributable to the ablated part and not to a second
      # fine-tuning run that happened to go differently.

      # No infilling loss. The remaining objective is the ordinary ZipVoice
      # loss, which over the generated span never asks the model to reconstruct
      # a missing band. If this matches the full method, the conditioning is
      # doing the work on its own and the loss term is not the contribution.
      ${PY} -m zipvoice.bin.train_zipvoice_anyband \
            "${common_args[@]}" \
            --finetune 1 --base-lr 0.001 \
            --num-iters ${frozen_iters} \
            --save-every-n ${save_every_n} --valid-interval ${valid_interval} \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --anyband-trainable anyband \
            --curriculum-steps ${frozen_iters} \
            --infill-loss-weight 0 \
            --exp-dir ${exp_dir}_no_infill

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${frozen_iters} --avg 2 \
            --model-name zipvoice_anyband \
            --exp-dir ${exp_dir}_no_infill

      # Fixed 8/4/4 schedule instead of the learned gate. This is the schedule
      # the design started from; the learned gate only pays for itself if it
      # beats it.
      ${PY} -m zipvoice.bin.train_zipvoice_anyband \
            "${common_args[@]}" \
            --finetune 1 --base-lr 0.001 \
            --num-iters ${frozen_iters} \
            --save-every-n ${save_every_n} --valid-interval ${valid_interval} \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --anyband-trainable anyband \
            --curriculum-steps ${frozen_iters} \
            --gate-fixed-schedule 1 \
            --exp-dir ${exp_dir}_fixed_gate

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${frozen_iters} --avg 2 \
            --model-name zipvoice_anyband \
            --exp-dir ${exp_dir}_fixed_gate
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Synthesise from band-limited prompts"
      # Scoring is stage 10 of run_bwe_libritts.sh, which reads these
      # directories; the stock arm it needs is produced there as well.
      # Only the proposed system is synthesised by default. The ablation arms
      # are listed here so that they can be added back with an override once the
      # main line has been reported.
      anyband_arms=${anyband_arms:-"${exp_dir}:${joint_iters}:anyband"}
      for arm in ${anyband_arms}; do
            dir=${arm%%:*}
            rest=${arm#*:}
            iters=${rest%%:*}
            name=${rest#*:}

            # Arms that were not trained are skipped rather than failing, so a
            # partial run still produces a scorable subset.
            if [ ! -f ${dir}/iter-${iters}-avg-2.pt ]; then
                  echo "Skipping ${name}: ${dir}/iter-${iters}-avg-2.pt not found"
                  continue
            fi
            cp conf/zipvoice_anyband.json ${dir}/model.json
            cp ${token_file} ${dir}/tokens.txt

            for rate in ${eval_rates}; do
                  ${PY} -m zipvoice.bin.infer_zipvoice \
                        --model-name zipvoice_anyband \
                        --model-dir ${dir}/ \
                        --checkpoint-name iter-${iters}-avg-2.pt \
                        --tokenizer ${tokenizer} \
                        --test-list ${eval_dir}/test_sr${rate}.tsv \
                        --res-dir ${results_dir}/${name}_sr${rate} \
                        --num-step 16
            done
      done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Score the AnyBand arms"
      # Scoring itself lives in run_bwe_libritts.sh; only the arm list differs,
      # so that stage is reused rather than duplicated. Arms that were never
      # synthesised are skipped there by the existing directory guard.
      # The reported comparison: released ZipVoice, the modular mel-to-mel extender
      # merged onto frozen released weights, and the proposed system.
      eval_systems="${eval_systems:-stock plugplay_frozen anyband}"             stage=10 stop_stage=10 eval_rates="${eval_rates}"             bash run_bwe_libritts.sh
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Adversarial joint fine-tuning"
      # An alternative to stage 3, run from the same stage-2 starting point so
      # the two are directly comparable.
      #
      # Two changes. The infilling term is down-weighted: it supervises the
      # prompt frames, which are discarded before vocoding, so it is an
      # auxiliary task whose gradient competed with the flow-matching loss at
      # roughly equal magnitude in stage 3. Its purpose is representational --
      # teach the decoder a low-band-to-high-band mapping that transfers to the
      # generated span -- and that does not require it to dominate.
      #
      # In its place the generated span gets an adversarial term. The
      # discriminators judge the endpoint estimate rather than the velocity,
      # only once the trajectory has passed the halfway point, and only over the
      # frames that actually reach the vocoder. This targets the failure stage 3
      # showed directly: output that satisfied a squared-error objective while
      # losing speaker detail and naturalness.
      adv_dir=${adv_dir:-${exp_dir}_adv}
      ${PY} -m zipvoice.bin.train_zipvoice_anyband \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.0005 \
            --num-iters ${joint_iters} \
            --save-every-n ${save_every_n} \
            --valid-interval ${valid_interval} \
            --checkpoint ${exp_dir}_frozen/iter-${frozen_iters}-avg-2.pt \
            --anyband-trainable decoder \
            --curriculum-rho 0.0 \
            --curriculum-steps 0 \
            --infill-loss-weight ${adv_infill_weight:-0.2} \
            --anyband-gan-start-step ${gan_start_step:-1000} \
            --anyband-adv-loss-weight ${adv_loss_weight:-0.1} \
            --anyband-fmatch-loss-weight ${fmatch_loss_weight:-1.0} \
            --anyband-adv-t-min ${adv_t_min:-0.5} \
            --exp-dir ${adv_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${joint_iters} --avg 2 \
            --model-name zipvoice_anyband \
            --exp-dir ${adv_dir}
fi
