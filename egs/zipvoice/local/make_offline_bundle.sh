#!/bin/bash

# Build a self-contained folder that trains and runs the dual-diffusion arm on a
# machine with no internet access.
#
# Nothing in the training or inference path reaches the network once the weights
# are on disk: the backbone checkpoint, its config and token table, and the Vocos
# vocoder are copied in, the recipe passes every one of them by path, and the
# run script exports the HuggingFace offline switches so a stray from_pretrained
# fails loudly instead of silently dialling out.
#
# Usage:
#   bash local/make_offline_bundle.sh [OUT_DIR]
#
# Environment:
#   ZIPVOICE_DATA   where the released checkpoint lives (default /c/zipvoice_data)
#   VOCOS_DIR       a directory holding config.yaml and pytorch_model.bin; by
#                   default the newest charactr/vocos-mel-24khz snapshot in the
#                   local HuggingFace cache
#   BRANCH_CKPT     a trained dual checkpoint to ship alongside the backbone,
#                   so the bundle can infer as well as train (optional)

export PYTHONUTF8=1
set -e
set -u
set -o pipefail

repo_root=$(cd ../.. && pwd)
data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
out_dir=${1:-${data_root}/dist/zipvoice_dual_offline}

src_zipvoice=${data_root}/download/zipvoice

if [ -z "${VOCOS_DIR:-}" ]; then
      # The cache keeps one directory per revision; take the most recent.
      cache=${HF_HOME:-${HOME}/.cache/huggingface}/hub
      VOCOS_DIR=$(ls -dt "${cache}"/models--charactr--vocos-mel-24khz/snapshots/*/ \
            2>/dev/null | head -1 || true)
fi
if [ -z "${VOCOS_DIR}" ] || [ ! -f "${VOCOS_DIR}/pytorch_model.bin" ]; then
      echo "No Vocos weights found. Set VOCOS_DIR to a directory holding" >&2
      echo "config.yaml and pytorch_model.bin (charactr/vocos-mel-24khz)." >&2
      exit 1
fi

echo "Writing ${out_dir}"
rm -rf "${out_dir}"
mkdir -p "${out_dir}/assets/zipvoice" "${out_dir}/assets/vocos-mel-24khz"

# Tracked files only, at HEAD: whatever is in the working tree but not committed
# is by definition not part of the version being shipped.
git -C "${repo_root}" archive HEAD zipvoice egs/zipvoice requirements.txt \
      | tar -x -C "${out_dir}"

cp "${src_zipvoice}/model.pt" "${out_dir}/assets/zipvoice/model.pt"
cp "${src_zipvoice}/model.json" "${out_dir}/assets/zipvoice/model.json"
cp "${src_zipvoice}/tokens.txt" "${out_dir}/assets/zipvoice/tokens.txt"
cp "${VOCOS_DIR}/config.yaml" "${out_dir}/assets/vocos-mel-24khz/config.yaml"
cp "${VOCOS_DIR}/pytorch_model.bin" \
      "${out_dir}/assets/vocos-mel-24khz/pytorch_model.bin"

if [ -n "${BRANCH_CKPT:-}" ]; then
      mkdir -p "${out_dir}/assets/dual"
      cp "${BRANCH_CKPT}" "${out_dir}/assets/dual/model.pt"
      cp "${src_zipvoice}/tokens.txt" "${out_dir}/assets/dual/tokens.txt"
      cp "$(dirname "${BRANCH_CKPT}")/model.json" "${out_dir}/assets/dual/model.json"
fi

cat > "${out_dir}/run_offline.sh" <<'EOS'
#!/bin/bash

# Entry point for the offline bundle. Sets every path the recipe would otherwise
# look up on HuggingFace, then hands over to the dual recipe.
#
#   bash run_offline.sh              # train (stage 1) through scoring (stage 3)
#   stage=2 stop_stage=2 bash run_offline.sh
#
# Expects the corpus and features under ${ZIPVOICE_DATA}, prepared as the
# README describes. Nothing here touches the network.

export PYTHONUTF8=1
set -e
set -u
set -o pipefail

bundle=$(cd "$(dirname "$0")" && pwd)

# Fail loudly rather than dial out: with these set, any from_pretrained or
# hf_hub_download that a path was not supplied for raises instead of fetching.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export ZIPVOICE_DATA=${ZIPVOICE_DATA:-${bundle}/data}
export ZIPVOICE_ASSETS=${bundle}/assets

cd "${bundle}/egs/zipvoice"
exec bash run_dual_libritts.sh "$@"
EOS
chmod +x "${out_dir}/run_offline.sh"

cat > "${out_dir}/README.md" <<'EOS'
# ZipVoice dual-diffusion, offline bundle

A frozen released ZipVoice plus a second flow-matching branch that writes only
the mel bins a band-limited prompt does not observe. Everything the training and
inference paths need is inside this folder; no step reaches the network.

## Layout

    zipvoice/                     the package
    egs/zipvoice/                 the recipe (run_dual_libritts.sh)
    assets/zipvoice/              the released backbone: model.pt, model.json, tokens.txt
    assets/vocos-mel-24khz/       the vocoder: config.yaml, pytorch_model.bin
    assets/dual/                  a trained dual checkpoint, if one was shipped
    run_offline.sh                entry point

## Install

    pip install -r requirements.txt
    pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Both are the only network steps, and they happen on a machine that has internet.
Mirror the wheels to the private server, or build a virtualenv elsewhere and copy
it across.

## Data

Set `ZIPVOICE_DATA` to a writable directory (it defaults to `data/` inside the
bundle) and place the corpus at `${ZIPVOICE_DATA}/download/LibriTTS`, then:

    cd egs/zipvoice
    bash local/prepare_libritts.sh

`prepare_libritts.sh` skips its download when `${dl_dir}/LibriTTS` already exists,
so a pre-copied corpus is used as is. Features land in
`${ZIPVOICE_DATA}/data/fbank`, which is what the trainer reads.

The evaluation set (band-limited prompts under `${ZIPVOICE_DATA}/data/bwe_eval`)
is built by `local/prepare_bwe_eval.py` and `local/prepare_testset_bwe.py` from
the same corpus. Scoring additionally needs the TTS evaluation models
(`k2-fsa/TTS_eval_models`) under `${ZIPVOICE_DATA}/download/tts_eval_models`;
copy that directory in if stage 3 is to be run, or stop at stage 2.

## Run

    bash run_offline.sh                        # train, synthesise, score
    stage=1 stop_stage=1 bash run_offline.sh   # train only

Training starts from `assets/zipvoice/model.pt` and trains only `branch.*`:
about 17 M parameters of 140 M. Peak allocated memory is roughly 7.5 GB at
`--max-duration 72` before the adversarial stage turns on.

## Verifying it really is offline

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash run_offline.sh

is what `run_offline.sh` already does. To prove it, run it on a host with no
route: any missed path raises `OfflineModeIsEnabled` rather than hanging.
EOS

echo "Done. Bundle at ${out_dir}"
du -sh "${out_dir}" 2>/dev/null || true
