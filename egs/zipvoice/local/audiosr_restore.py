#!/usr/bin/env python3
"""Restore evaluation prompts with the released AudioSR checkpoint.

This is `nvsr_restore.py` with a different extender behind the same contract: a
test list goes in, a test list pointing at restored prompts comes out, and stock
ZipVoice is then conditioned on those. Nothing here is trained in this
repository -- the weights are the ones published with AudioSR, downloaded on
first use.

AudioSR is a latent diffusion model working at 48 kHz, so a prompt recorded at 8
or 16 kHz is restored there and resampled to the 24 kHz ZipVoice expects. Its own
post-processing is kept: `LatentDiffusion.postprocessing` splices the observed low
band back over the restored one in the STFT domain, which is the same
configuration NVSR was measured in, so the two cascades differ only in the
extender.

**This script does not run in the main environment.** AudioSR 0.0.7 pins
`numpy<=1.23.5`, `librosa==0.9.2` and `transformers==4.30.2`; installing that
would break the evaluation models the rest of the study scores with. It runs in
an overlay virtualenv instead, created with `--system-site-packages` over the
main one so torch is shared rather than downloaded twice:

    python -m venv --system-site-packages /c/zipvoice_data/venv_audiosr
    /c/zipvoice_data/venv_audiosr/Scripts/python -m pip install --no-deps audiosr==0.0.7
    /c/zipvoice_data/venv_audiosr/Scripts/python -m pip install \
        torchlibrosa progressbar ftfy timm chardet unidecode

Nothing from `zipvoice` is imported here, so running under a different
interpreter costs nothing.

Usage:

    /c/zipvoice_data/venv_audiosr/Scripts/python egs/zipvoice/local/audiosr_restore.py \
        --test-list  /c/zipvoice_data/data/seedtts_en_bwe/test_sr8000.tsv \
        --out-list   /c/zipvoice_data/data/seedtts_en_bwe/test_audiosr_sr8000.tsv \
        --prompt-dir /c/zipvoice_data/data/seedtts_en_bwe/prompts/audiosr_sr8000 \
        --timing-path /c/zipvoice_data/exp/results/audiosr_time_seedtts_en_8000.tsv
"""

import argparse
import csv
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# torchlibrosa calls librosa.util.pad_center(window, n_fft) positionally, which
# librosa made keyword-only in 0.10. AudioSR pins librosa 0.9.2, but that has no
# wheel for the Python this runs on, so the call is adapted rather than the
# library pinned back.
_pad_center = librosa.util.pad_center


def _pad_center_compat(data, size=None, *args, **kwargs):
    if size is not None:
        kwargs["size"] = size
    return _pad_center(data, *args, **kwargs)


librosa.util.pad_center = _pad_center_compat

from audiosr import build_model, super_resolution  # noqa: E402  (shim above)

AUDIOSR_SR = 48000


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument(
        "--timing-path",
        type=Path,
        default=None,
        help="""Where to write per-prompt restoration wall time. The cascade's
        real-time factor is the extender's cost plus synthesis, and only the
        first is measured here.""",
    )
    parser.add_argument("--model-name", type=str, default="basic")
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="AudioSR's own default since its 2023-09-16 release.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-rate", type=int, default=24000)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    args.prompt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_list, encoding="utf-8") as handle:
        # QUOTE_NONE: the transcripts contain bare double quotes, which the
        # default dialect reads as field quoting -- it silently merges rows and
        # strips the marks.
        rows = [
            row
            for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            if row
        ]

    model = build_model(model_name=args.model_name, device="auto")

    timings = []
    restored: dict = {}
    with open(args.out_list, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
        )
        for i, row in enumerate(rows):
            source = Path(row[2])
            out_path = args.prompt_dir / f"{source.stem}.wav"
            # Prompts repeat across rows, and a diffusion pass is expensive
            # enough that restoring one twice would be the dominant waste.
            # An existing file also makes an interrupted run resumable.
            if source not in restored:
                if out_path.is_file():
                    restored[source] = out_path
                else:
                    start = time.time()
                    y = np.squeeze(
                        super_resolution(
                            model,
                            str(source),
                            seed=args.seed,
                            guidance_scale=args.guidance_scale,
                            ddim_steps=args.ddim_steps,
                        )
                    )
                    elapsed = time.time() - start
                    y = librosa.resample(
                        y, orig_sr=AUDIOSR_SR, target_sr=args.sampling_rate
                    )
                    sf.write(out_path, y, args.sampling_rate)
                    restored[source] = out_path
                    timings.append(
                        (
                            source.stem,
                            f"{len(y) / args.sampling_rate:.3f}",
                            f"{elapsed:.3f}",
                        )
                    )
                    print(
                        f"[{i + 1}/{len(rows)}] {out_path} in {elapsed:.1f}s",
                        flush=True,
                    )
            writer.writerow([row[0], row[1], str(restored[source]), row[3]])

    if args.timing_path is not None and timings:
        args.timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.timing_path, "w", encoding="utf-8", newline="") as handle:
            out = csv.writer(handle, delimiter="\t", lineterminator="\n")
            out.writerow(["wav_name", "seconds_audio", "seconds_wall"])
            out.writerows(timings)
        total_audio = sum(float(t[1]) for t in timings)
        total_wall = sum(float(t[2]) for t in timings)
        print(
            f"Restored {len(timings)} prompts, {total_audio:.1f}s of audio in "
            f"{total_wall:.1f}s: real-time factor {total_wall / total_audio:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
