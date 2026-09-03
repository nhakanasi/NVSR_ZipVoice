#!/bin/bash

# This script fine-tunes a pre-trained ZipVoice into ZipVoiceBWE: the same TTS
# model with a mel bandwidth extender in front of the speech condition, so that
# a prompt recorded at any sampling rate can be used.
#
# The extender is trained on band-limited copies of the training mel, with the
# reconstruction, adversarial and feature-matching losses of AnyBand
# (arXiv 2608.00572) plus ZipVoice's own flow-matching loss, which is what makes
# the restored mel useful to the TTS rather than merely plausible to look at.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=8

# Number of jobs for data preparation
nj=20

# Whether the language of training data is one of Chinese and English
is_zh_en=1

# Language identifier, used when language is not Chinese or English
# see https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
lang=default

if [ $is_zh_en -eq 1 ]; then
      tokenizer=emilia
else
      tokenizer=espeak
      [ "$lang" = "default" ] && { echo "Error: lang is not set!" >&2; exit 1; }
fi

# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=20

# Download directory for pre-trained models
download_dir=download/

exp_dir=exp/zipvoice_bwe

# Iteration counts of the three training stages.
pretrain_iters=20000
frozen_iters=100000
joint_iters=60000

# See run_finetune.sh for the format of the TSV files expected here.
for subset in train dev;do
      file_path=data/raw/custom_${subset}.tsv
      [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
done

### Prepare the training data (1 - 4)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for custom dataset from tsv files"

      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path data/raw/custom_${subset}.tsv \
                  --prefix custom-bwe \
                  --subset raw_${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/custom-bwe_cuts_raw_${subset}.jsonl.gz \
                  --output-file data/manifests/custom-bwe_cuts_${subset}.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --lang ${lang}
      done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
      echo "Stage 3: Compute Fbank for custom dataset"
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset custom-bwe \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download the pre-trained ZipVoice model and tokens file"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
      # The model config comes from conf/zipvoice_bwe.json rather than the
      # downloaded model.json: it is the same ZipVoice architecture plus the
      # "bwe" and "bwe_disc" blocks.
fi

### Training (5 - 7)

# Arguments shared by all three training stages.
common_args=(
      --world-size 4
      --use-fp16 1
      --max-duration 500
      --max-len ${max_len}
      --model-config conf/zipvoice_bwe.json
      --tokenizer ${tokenizer}
      --lang ${lang}
      --token-file ${download_dir}/zipvoice/tokens.txt
      --dataset custom
      --train-manifest data/fbank/custom-bwe_cuts_train.jsonl.gz
      --dev-manifest data/fbank/custom-bwe_cuts_dev.jsonl.gz
)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Pre-train the bandwidth extender on reconstruction alone"
      # The public NVSR weights are 44.1 kHz on a 128-bin mel with a different
      # filterbank, so they do not transfer to ZipVoice's 100-bin 24 kHz mel.
      # This stage is the substitute: the extender alone, no flow-matching
      # gradient and no discriminators, on the Easy-to-Balanced cutoff
      # curriculum.
      python3 -m zipvoice.bin.train_zipvoice_bwe \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.001 \
            --num-iters ${pretrain_iters} \
            --save-every-n 5000 \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --bwe-freeze-zipvoice 1 \
            --bwe-fm-loss-weight 0 \
            --bwe-gan-start-step -1 \
            --bwe-curriculum-steps ${pretrain_iters} \
            --exp-dir ${exp_dir}_pretrain

      python3 -m zipvoice.bin.generate_averaged_model \
            --iter ${pretrain_iters} \
            --avg 2 \
            --model-name zipvoice_bwe \
            --exp-dir ${exp_dir}_pretrain
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Train the extender against the frozen TTS"
      # ZipVoice stays frozen, so the flow-matching gradient reaching the
      # extender tells it what the TTS actually needs from the prompt mel,
      # and the TTS cannot adapt to a bad extender instead.
      python3 -m zipvoice.bin.train_zipvoice_bwe \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.0005 \
            --num-iters ${frozen_iters} \
            --save-every-n 5000 \
            --checkpoint ${exp_dir}_pretrain/iter-${pretrain_iters}-avg-2.pt \
            --bwe-freeze-zipvoice 1 \
            --bwe-fm-loss-weight 1.0 \
            --bwe-gan-start-step -1 \
            --bwe-curriculum-steps ${frozen_iters} \
            --exp-dir ${exp_dir}_frozen

      python3 -m zipvoice.bin.generate_averaged_model \
            --iter ${frozen_iters} \
            --avg 2 \
            --model-name zipvoice_bwe \
            --exp-dir ${exp_dir}_frozen
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Joint adversarial fine-tuning"
      # Everything trains now, at a low learning rate, with all three
      # discriminators. Cutoffs are sampled uniformly here: per AnyBand the
      # curriculum belongs to the pre-adversarial stages only, which is what
      # --bwe-curriculum-steps 0 selects.
      python3 -m zipvoice.bin.train_zipvoice_bwe \
            "${common_args[@]}" \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters ${joint_iters} \
            --save-every-n 5000 \
            --checkpoint ${exp_dir}_frozen/iter-${frozen_iters}-avg-2.pt \
            --bwe-freeze-zipvoice 0 \
            --bwe-fm-loss-weight 1.0 \
            --bwe-gan-start-step 0 \
            --bwe-adv-loss-weight 0.1 \
            --bwe-feat-match-loss-weight 2.0 \
            --bwe-disc-lr 0.0002 \
            --bwe-curriculum-steps 0 \
            --exp-dir ${exp_dir}

      python3 -m zipvoice.bin.generate_averaged_model \
            --iter ${joint_iters} \
            --avg 2 \
            --model-name zipvoice_bwe \
            --exp-dir ${exp_dir}
fi

### Inference (8)

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Inference with prompts at their original sampling rate"
      # The prompt wav no longer has to be 24 kHz: infer_zipvoice reads the
      # file's own sampling rate and restores the mel above its Nyquist
      # frequency before conditioning.
      cp conf/zipvoice_bwe.json ${exp_dir}/model.json
      cp ${download_dir}/zipvoice/tokens.txt ${exp_dir}/tokens.txt

      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_bwe \
            --model-dir ${exp_dir}/ \
            --checkpoint-name iter-${joint_iters}-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_bwe \
            --num-step 16
fi
