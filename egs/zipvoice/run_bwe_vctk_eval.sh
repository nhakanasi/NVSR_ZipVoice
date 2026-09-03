#!/bin/bash

# Evaluate the trained ZipVoiceBWE checkpoint on VCTK, a corpus it has never
# seen.
#
# run_bwe_libritts.sh trains on LibriTTS and evaluates on LibriTTS dev-clean.
# That measures the extender, but it cannot rule out the extender and the
# fine-tuned TTS weights having simply adapted to LibriTTS -- same source
# material, same recording chain, same reading style. VCTK changes all three:
# different speakers, a head-mounted DPA in a hemi-anechoic room rather than
# audiobook recordings, many non-American accents, and short read newspaper
# sentences.
#
# The arms are the same four as stage 9/10 of the LibriTTS recipe, so the two
# result tables can be read side by side. The decisive contrast is bwe against
# bwe_bypass_full: same weights, same prompt, extender on or off.

export PYTHONUTF8=1

set -e
set -u
set -o pipefail

if python3 -c "" >/dev/null 2>&1; then PY=python3; else PY=python; fi

if ${PY} -c "import sys; sys.exit(0 if sys.platform == 'win32' else 1)"; then
      export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
else
      export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"
fi

stage=${stage:-1}
stop_stage=${stop_stage:-3}

eval_utts=${eval_utts:-76}
eval_rates=${eval_rates:-"8000 16000 22050 24000"}
eval_rates_csv=$(echo ${eval_rates} | tr ' ' ',')

data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
download_dir=${data_root}/download
exp_root=${data_root}/exp
exp_dir=${exp_root}/zipvoice_bwe
eval_dir=${data_root}/data/vctk_bwe_eval
results_dir=${exp_root}/vctk_results
eval_model_dir=${download_dir}/tts_eval_models

tokenizer=emilia
token_file=${download_dir}/zipvoice/tokens.txt
joint_iters=${joint_iters:-15000}

# VCTK 0.92. The prepare script reads members over HTTP range requests, so this
# stays a URL rather than an 11.7 GB download; only the ~80 prompt recordings
# and their transcripts are actually transferred. Point it at a local
# VCTK-Corpus-0.92.zip instead if you already have one.
vctk_zip=${vctk_zip:-https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Build the VCTK evaluation set"
      ${PY} local/prepare_vctk_bwe_eval.py \
            --vctk-zip "${vctk_zip}" \
            --out-dir ${eval_dir} \
            --num-utts ${eval_utts} \
            --rates "${eval_rates_csv}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Synthesise every arm from the VCTK prompts"
      cp conf/zipvoice_bwe.json ${exp_dir}/model.json
      cp ${token_file} ${exp_dir}/tokens.txt

      for rate in ${eval_rates}; do
            # Directory names match the LibriTTS recipe so the two result
            # tables line up: the unbypassed arm is plain `bwe`.
            for bypass in none resunet full; do
                  if [ ${bypass} = none ]; then
                        arm=bwe
                  else
                        arm=bwe_bypass_${bypass}
                  fi
                  ${PY} -m zipvoice.bin.infer_zipvoice \
                        --model-name zipvoice_bwe \
                        --model-dir ${exp_dir}/ \
                        --checkpoint-name iter-${joint_iters}-avg-2.pt \
                        --tokenizer ${tokenizer} \
                        --test-list ${eval_dir}/test_sr${rate}.tsv \
                        --res-dir ${results_dir}/${arm}_sr${rate} \
                        --bwe-bypass ${bypass} \
                        --num-step 16
            done

            # The released checkpoint, which saw neither the extender nor
            # LibriTTS fine-tuning.
            ${PY} -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice \
                  --tokenizer ${tokenizer} \
                  --test-list ${eval_dir}/test_sr${rate}.tsv \
                  --res-dir ${results_dir}/stock_sr${rate} \
                  --num-step 16
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Score every condition"
      # Scored against test_score.tsv, which points at the original 24 kHz
      # prompt rather than the degraded one, so the score measures the synthesis
      # instead of the degradation.
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
