#!/bin/bash

# Fine-tune the released ZipVoice checkpoint into ZipVoiceBWE on LibriTTS.
#
# LibriTTS rather than LibriSpeech: the bandwidth extender is supervised against
# the full-band mel of the training audio, so the corpus has to actually contain
# the band the extender must learn to restore. LibriSpeech is 16 kHz, which
# leaves mel bins above 8 kHz empty in the target and teaches the extender to
# emit the log floor. LibriTTS is the same source material at 24 kHz.


# Emilia tokens are IPA. lhotse writes manifests with the locale codec, which on
# a Windows machine is cp1252 and cannot encode them, so force UTF-8 mode.
export PYTHONUTF8=1

set -e
set -u
set -o pipefail

# Windows ships a python3.exe App Execution Alias that is on PATH but only
# prints an install prompt, so probe by running it, not by command -v.
if python3 -c "" >/dev/null 2>&1; then PY=python3; else PY=python; fi

# Add the project root to PYTHONPATH. The separator is ";" for a Windows
# interpreter and ":" elsewhere, and a Windows interpreter needs a native path,
# so ask the interpreter itself rather than assuming POSIX.
if ${PY} -c "import sys; sys.exit(0 if sys.platform == 'win32' else 1)"; then
      export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
else
      export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"
fi

stage=${stage:-1}
stop_stage=${stop_stage:-10}

# Evaluation set (stages 8-10). dev-clean has 40 speakers, too few to reach a
# useful sample size at one pair each, so more than one pair per speaker is
# allowed. 24000 is the control condition and must not regress.
eval_utts=${eval_utts:-80}
eval_per_speaker=${eval_per_speaker:-2}
eval_rates=${eval_rates:-"8000 16000 22050 24000"}
eval_rates_csv=$(echo ${eval_rates} | tr ' ' ',')

# Set wandb=1 and export WANDB_API_KEY (or run `wandb login`) to mirror the
# tensorboard metrics to Weights & Biases. Never put the key in this file.
wandb=${wandb:-0}
wandb_project=${wandb_project:-zipvoice-bwe}

# recon_loss_type=spectral swaps the L1 on the restored band for the
# LSD + ILD + NDL objective of HRTFformer (arXiv 2510.01891). Those terms are
# in dB and sum to ~55 where the L1 sits at ~0.1, so the weight drops with them.
recon_loss_type=${recon_loss_type:-l1}
if [ "${recon_loss_type}" = "spectral" ]; then
      recon_loss_weight=${recon_loss_weight:-0.002}
else
      recon_loss_weight=${recon_loss_weight:-1.0}
fi

# Corpus, features and checkpoints live outside the repository: they are tens of
# gigabytes and the repository may sit on a synced or streamed drive.
data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
download_dir=${data_root}/download
manifest_dir=${data_root}/data/manifests
fbank_dir=${data_root}/data/fbank
exp_root=${data_root}/exp
eval_dir=${data_root}/data/bwe_eval
results_dir=${exp_root}/results
# Speaker-verification, ASR and MOS checkpoints for stage 10. These are several
# gigabytes and are not fetched by stage 1; stage 8 pulls them.
eval_model_dir=${download_dir}/tts_eval_models

# Subsets to use. dev-clean is the validation set expected by --dataset libritts.
train_subsets="train-clean-360"

nj=8
max_len=20

# The tokenizer and token file have to be the ones the released checkpoint was
# trained with, otherwise the text embedding table does not match the weights.
tokenizer=emilia
token_file=${download_dir}/zipvoice/tokens.txt

# Single GPU. Measured on a 16 GB card: 9.7 GB peak at --max-duration 60 with
# 20 s utterances, so 72 leaves room for fp16 grad-scale spikes. Raise it on a
# larger card; on Windows an over-large value does not OOM, it spills to host
# memory and quietly halves throughput.
world_size=1
max_duration=72

exp_dir=${exp_root}/zipvoice_bwe

pretrain_iters=8000
frozen_iters=30000
joint_iters=15000

# Checkpointing cadence for stages 6 and 7. Each checkpoint is about 2 GB on
# disk and is what --resume-from names, so this also sets how much progress an
# interrupted run can lose.
save_every_n=${save_every_n:-1000}

# Validation is separately gated because it now runs the discriminators as
# well, which cost roughly six times a reconstruction-only pass. Checkpointing
# every 1000 batches while validating every 2000 keeps the resume granularity
# without spending a third of the run on the dev set.
valid_interval=${valid_interval:-2000}

# Point resume_from at a checkpoint-{batch}.pt to continue a stage 7 run that
# was interrupted. It restores the step count, both optimizers, the
# discriminators and the Weights & Biases run id, so the resumed run extends
# the original rather than starting a second one beside it.
resume_from=${resume_from:-}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Download LibriTTS (24 kHz) and the pre-trained ZipVoice"
      mkdir -p ${download_dir}
      if [ ! -d ${download_dir}/LibriTTS ]; then
            parts=""
            for subset in ${train_subsets} dev-clean; do
                  parts="${parts} -p ${subset}"
            done
            lhotse download libritts ${parts} ${download_dir}
      fi

      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      # huggingface-cli was removed in huggingface_hub 1.x and now only prints
      # a deprecation notice while exiting non-zero; hf is its replacement.
      for file in model.pt tokens.txt; do
            hf download k2-fsa/ZipVoice zipvoice/${file} \
                  --local-dir ${download_dir}
      done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Prepare LibriTTS manifests"
      mkdir -p ${manifest_dir}
      parts=""
      for subset in ${train_subsets} dev-clean; do
            parts="${parts} -p ${subset}"
      done
      lhotse prepare libritts --num-jobs ${nj} ${parts} \
            ${download_dir}/LibriTTS ${manifest_dir}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Compute 24 kHz fbank features"
      mkdir -p ${fbank_dir}
      for subset in ${train_subsets} dev-clean; do
            ${PY} -m zipvoice.bin.compute_fbank \
                  --source-dir ${manifest_dir} \
                  --dest-dir ${fbank_dir} \
                  --dataset libritts \
                  --subset ${subset} \
                  --sampling-rate 24000 \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Add Emilia tokens and build the shuffled training manifest"
      # Tokenizing here rather than on the fly: the training loop tokenizes in
      # the main process, and espeak on every batch throttles the GPU.
      for subset in ${train_subsets} dev-clean; do
            ${PY} -m zipvoice.bin.prepare_tokens \
                  --input-file ${fbank_dir}/libritts_cuts_${subset}.jsonl.gz \
                  --output-file ${fbank_dir}/libritts_tok_cuts_${subset}.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --num-jobs ${nj}
      done

      # --dataset libritts reads these two exact file names.
      for subset in ${train_subsets}; do
            gunzip -c ${fbank_dir}/libritts_tok_cuts_${subset}.jsonl.gz
      done | shuf | gzip -c > ${fbank_dir}/libritts_cuts_train-all-shuf.jsonl.gz
      cp ${fbank_dir}/libritts_tok_cuts_dev-clean.jsonl.gz \
            ${fbank_dir}/libritts_cuts_dev-clean.jsonl.gz
fi

common_args=(
      --world-size ${world_size}
      --use-fp16 1
      --max-duration ${max_duration}
      --max-len ${max_len}
      --model-config conf/zipvoice_bwe.json
      --tokenizer ${tokenizer}
      --token-file ${token_file}
      --dataset libritts
      --manifest-dir ${fbank_dir}
      --wandb ${wandb}
      --wandb-project ${wandb_project}
      --bwe-recon-loss-type ${recon_loss_type}
      --bwe-recon-loss-weight ${recon_loss_weight}
)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Pre-train the bandwidth extender on reconstruction alone"
      ${PY} -m zipvoice.bin.train_zipvoice_bwe \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.001 \
            --num-iters ${pretrain_iters} \
            --save-every-n 2000 \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --bwe-freeze-zipvoice 1 \
            --bwe-fm-loss-weight 0 \
            --bwe-gan-start-step -1 \
            --bwe-curriculum-steps ${pretrain_iters} \
            --exp-dir ${exp_dir}_pretrain

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${pretrain_iters} --avg 2 \
            --model-name zipvoice_bwe \
            --exp-dir ${exp_dir}_pretrain
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Train the extender against the frozen TTS"
      ${PY} -m zipvoice.bin.train_zipvoice_bwe \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.0005 \
            --num-iters ${frozen_iters} \
            --save-every-n ${save_every_n} \
            --valid-interval ${valid_interval} \
            --checkpoint ${exp_dir}_pretrain/iter-${pretrain_iters}-avg-2.pt \
            --bwe-freeze-zipvoice 1 \
            --bwe-fm-loss-weight 1.0 \
            --bwe-gan-start-step -1 \
            --bwe-curriculum-steps ${frozen_iters} \
            --exp-dir ${exp_dir}_frozen

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${frozen_iters} --avg 2 \
            --model-name zipvoice_bwe \
            --exp-dir ${exp_dir}_frozen
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Joint adversarial fine-tuning"
      if [ -n "${resume_from}" ]; then
            start_args=(--resume-from "${resume_from}")
      else
            start_args=(--checkpoint ${exp_dir}_frozen/iter-${frozen_iters}-avg-2.pt)
      fi
      ${PY} -m zipvoice.bin.train_zipvoice_bwe \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters ${joint_iters} \
            --save-every-n ${save_every_n} \
            --valid-interval ${valid_interval} \
            "${start_args[@]}" \
            --bwe-freeze-zipvoice 0 \
            --bwe-fm-loss-weight 1.0 \
            --bwe-gan-start-step 0 \
            --bwe-adv-loss-weight 0.1 \
            --bwe-feat-match-loss-weight 2.0 \
            --bwe-disc-lr 0.0002 \
            --bwe-curriculum-steps 0 \
            --exp-dir ${exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${joint_iters} --avg 2 \
            --model-name zipvoice_bwe \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Build the band-limited evaluation set"
      # Speaker-verification, ASR and MOS checkpoints for stage 10, in one
      # repository. Several gigabytes, so it is skipped once present.
      if [ ! -d ${eval_model_dir} ]; then
            hf download k2-fsa/TTS_eval_models --local-dir ${eval_model_dir}
      fi

      # Held-out prompts resampled down to each target rate and written out at
      # that rate, so inference sees exactly what a user supplying a low-rate
      # file would give it. 24000 is the undegraded control.
      ${PY} local/prepare_bwe_eval.py \
            --cuts ${fbank_dir}/libritts_cuts_dev-clean.jsonl.gz \
            --out-dir ${eval_dir} \
            --num-utts ${eval_utts} \
            --max-per-speaker ${eval_per_speaker} \
            --rates "${eval_rates_csv}"
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
      echo "Stage 9: Synthesise from band-limited prompts, extender vs stock"
      cp conf/zipvoice_bwe.json ${exp_dir}/model.json
      cp ${token_file} ${exp_dir}/tokens.txt

      for rate in ${eval_rates}; do
            # The extender under test.
            ${PY} -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice_bwe \
                  --model-dir ${exp_dir}/ \
                  --checkpoint-name iter-${joint_iters}-avg-2.pt \
                  --tokenizer ${tokenizer} \
                  --test-list ${eval_dir}/test_sr${rate}.tsv \
                  --res-dir ${results_dir}/bwe_sr${rate} \
                  --num-step 16

            # Same fine-tuned weights, extender partly or wholly switched
            # off. The stock arm below differs from the bwe arm in two ways at
            # once (extender present, and 15000 steps of LibriTTS fine-tuning),
            # so it cannot attribute a result to either. These two arms hold
            # the weights fixed and vary only the extender:
            #   resunet  dead band re-floored, ResUNet never predicts into it
            #   full     prompt mel untouched, the same input stock receives
            for bypass in resunet full; do
                  ${PY} -m zipvoice.bin.infer_zipvoice \
                        --model-name zipvoice_bwe \
                        --model-dir ${exp_dir}/ \
                        --checkpoint-name iter-${joint_iters}-avg-2.pt \
                        --tokenizer ${tokenizer} \
                        --test-list ${eval_dir}/test_sr${rate}.tsv \
                        --res-dir ${results_dir}/bwe_bypass_${bypass}_sr${rate} \
                        --bwe-bypass ${bypass} \
                        --num-step 16
            done

            # Stock ZipVoice on the same degraded prompts. This is the number
            # the extender has to beat; without it a good-looking BWE score
            # says nothing, because flow matching may already cope with a
            # partial-bandwidth prompt on its own.
            ${PY} -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice \
                  --tokenizer ${tokenizer} \
                  --test-list ${eval_dir}/test_sr${rate}.tsv \
                  --res-dir ${results_dir}/stock_sr${rate} \
                  --num-step 16
      done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
      echo "Stage 10: Score every condition"
      # Every condition is scored against test_score.tsv, which points at the
      # original 24 kHz prompt rather than the degraded one. Scoring against a
      # degraded reference would measure the degradation instead of the
      # synthesis, and would flatter every system equally.
      for system in bwe bwe_bypass_resunet bwe_bypass_full stock; do
            for rate in ${eval_rates}; do
                  wav_path=${results_dir}/${system}_sr${rate}
                  [ -d "${wav_path}" ] || continue
                  echo "--- ${system} @ ${rate} Hz ---"

                  ${PY} -m zipvoice.eval.speaker_similarity.sim \
                        --wav-path ${wav_path} \
                        --test-list ${eval_dir}/test_score.tsv \
                        --model-dir ${eval_model_dir}

                  ${PY} -m zipvoice.eval.wer.seedtts \
                        --wav-path ${wav_path} \
                        --test-list ${eval_dir}/test_score.tsv \
                        --decode-path ${wav_path}/decode.txt \
                        --model-dir ${eval_model_dir} \
                        --lang en

                  ${PY} -m zipvoice.eval.mos.utmos \
                        --wav-path ${wav_path} \
                        --model-dir ${eval_model_dir}

                  # DNSMOS P.835 splits quality into speech distortion,
                  # background intrusiveness and overall, which UTMOS's single
                  # number cannot separate. Per-file scores are kept so
                  # conditions can be compared with a paired test.
                  ${PY} -m zipvoice.eval.mos.dnsmos \
                        --wav-path ${wav_path} \
                        --model-dir ${eval_model_dir} \
                        --score-path ${wav_path}/dnsmos.tsv
            done
      done
fi
