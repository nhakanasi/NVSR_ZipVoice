#!/bin/bash
# Real-time factor for the three reported systems, measured by re-running the
# ordinary inference path and reading the per-sentence RTF that
# infer_zipvoice already logs. Timing the real path rather than a synthetic
# harness keeps the number comparable with what the evaluation actually ran.
#
# Waveforms go to a scratch directory and are discarded: the scored results
# must not be overwritten, because synthesis is stochastic and the scores
# would no longer match the audio they were computed from.
set -e
cd "$(dirname "$0")/.."

export PYTHONUTF8=1
if python3 -c "" >/dev/null 2>&1; then PY=python3; else PY=python; fi
# Same PYTHONPATH construction as the run scripts: a Windows-shaped path with
# ";" separators under git-bash, a POSIX one elsewhere.
if [ "$(uname -o 2>/dev/null)" = "Msys" ]; then
      export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
else
      export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"
fi
data_root=${data_root:-/c/zipvoice_data}
eval_dir=${eval_dir:-${data_root}/data/bwe_eval}
scratch=${scratch:-${data_root}/exp/rtf_scratch}
tokenizer=${tokenizer:-emilia}
rates=${rates:-"8000 16000 22050"}
anyband_dir=${anyband_dir:-${data_root}/exp/zipvoice_anyband}
anyband_ckpt=${anyband_ckpt:-iter-9000-avg-2.pt}
plugplay_dir=${plugplay_dir:-${data_root}/exp/zipvoice_bwe_plugplay_frozen}
plugplay_ckpt=${plugplay_ckpt:-iter-30000-avg-2.pt}

mkdir -p ${scratch}

for rate in ${rates}; do
      list=${eval_dir}/test_sr${rate}.tsv

      ${PY} -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --tokenizer ${tokenizer} \
            --test-list ${list} \
            --res-dir ${scratch}/stock_sr${rate} \
            --num-step 16

      ${PY} -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_bwe \
            --model-dir ${plugplay_dir}/ \
            --checkpoint-name ${plugplay_ckpt} \
            --tokenizer ${tokenizer} \
            --test-list ${list} \
            --res-dir ${scratch}/plugplay_frozen_sr${rate} \
            --num-step 16

      ${PY} -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_anyband \
            --model-dir ${anyband_dir}/ \
            --checkpoint-name ${anyband_ckpt} \
            --tokenizer ${tokenizer} \
            --test-list ${list} \
            --res-dir ${scratch}/anyband_sr${rate} \
            --num-step 16
done
