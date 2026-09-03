#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mel-cepstral distortion between synthesised speech and a natural reference.

Every quality number elsewhere in this study is a learned MOS predictor, and
both of the ones used run at 16 kHz internally, so neither can see the band the
extender is supposed to restore. MCD is a signal-level distance instead of a
learned opinion: it compares WORLD spectral envelopes directly, at whatever rate
the audio carries, so it observes the full band and cannot be talked into a good
score by output that merely sounds plausible.

What it is not is a fair measure of a zero-shot TTS system in absolute terms.
The reference is a different rendition of the same sentence by the same speaker,
with its own prosody and phone durations, so a large part of the distance is
timing and delivery that the model was never asked to reproduce. Dynamic time
warping removes the duration component but not the delivery one. The number is
therefore useful for ranking arms that were given identical text, identical
speakers and identical references -- which is exactly the comparison this study
makes -- and misleading if quoted against MCD figures from a resynthesis task.

Convention followed here, which is the common one in the TTS literature: order
34 mel-cepstra from the WORLD spectral envelope, coefficient zero dropped so
that gain differences do not enter, Euclidean distance accumulated along a DTW
path, scaled by 10 / ln(10) * sqrt(2).

Usage:

    python egs/zipvoice/local/mcd_eval.py \
        --wav-path exp/results/bwe_sr8000 \
        --ref-path data/bwe_eval/references \
        --score-path mcd/libritts_bwe_sr8000.tsv
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pysptk
import pyworld
import soundfile as sf

# WORLD's default frame period, in milliseconds.
FRAME_PERIOD = 5.0

# All-pole frequency warping coefficient. The usual values are 0.42 at 16 kHz
# and 0.455 at 22.05 kHz; 0.466 is the corresponding value at 24 kHz, which is
# the rate everything in this recipe runs at.
ALPHA_24K = 0.466

MCD_SCALE = 10.0 / np.log(10.0) * np.sqrt(2.0)


def mel_cepstrum(
    path: Path, order: int, alpha: float, sample_rate: int
) -> np.ndarray:
    """Order-`order` mel-cepstra of one file, coefficient zero dropped."""
    audio, rate = sf.read(str(path))
    # The rest of the evaluation suite resamples to whatever rate its model
    # wants. MCD must not: the warping coefficient is tied to the rate, and two
    # envelopes computed at different rates are not comparable. Refuse instead.
    if rate != sample_rate:
        raise ValueError(
            f"{path} is at {rate} Hz, expected {sample_rate} Hz; alpha={alpha} "
            f"is only correct at the expected rate"
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    # DIO with the stonemask refinement rather than Harvest: an order of
    # magnitude faster, and the F0 track is only used to drive CheapTrick and to
    # mark voiced frames, neither of which is sensitive to the difference.
    f0, t = pyworld.dio(audio, rate, frame_period=FRAME_PERIOD)
    f0 = pyworld.stonemask(audio, f0, t, rate)
    spectrum = pyworld.cheaptrick(audio, f0, t, rate)
    mc = pysptk.sp2mc(spectrum, order=order, alpha=alpha)
    # Frames where WORLD found no periodicity are silence or noise; including
    # them measures how the two recordings pad their sentences rather than how
    # they pronounce it.
    voiced = f0 > 0
    if voiced.sum() < 2:
        return mc[:, 1:]
    return mc[voiced][:, 1:]


def dtw_mean_distance(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Mean Euclidean distance along the DTW path between two frame sequences."""
    # (len(a), len(b)) pairwise distances, then the standard accumulation. The
    # sequences here are a few hundred frames, so the quadratic cost is not
    # worth avoiding with a band or a lower bound.
    diff = a[:, None, :] - b[None, :, :]
    local = np.sqrt((diff * diff).sum(axis=-1))

    n, m = local.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row = local[i - 1]
        prev, cur = acc[i - 1], acc[i]
        for j in range(1, m + 1):
            cur[j] = row[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])

    # Walk the path back to recover its length, which is what the total has to
    # be divided by: a longer alignment accumulates more distance for free.
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        choices = (acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
        best = int(np.argmin(choices))
        if best == 0:
            i -= 1
        elif best == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    steps += i + j
    return float(acc[n, m] / steps), steps


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--wav-path",
        type=str,
        required=True,
        help="Directory of synthesised wavs.",
    )
    parser.add_argument(
        "--ref-path",
        type=str,
        required=True,
        help="""Directory of natural references, one per synthesised file and
        named identically. Build it with local/prepare_mcd_references.py.
        """,
    )
    parser.add_argument(
        "--score-path",
        type=str,
        default=None,
        help="Optional TSV to write per-utterance scores to, for paired tests.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Rate both the synthesised and the reference audio must carry.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=34,
        help="Mel-cepstrum order.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_24K,
        help="Frequency warping coefficient; the default is for 24 kHz.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )

    wav_path = Path(args.wav_path)
    ref_path = Path(args.ref_path)
    logging.info(f"Calculating MCD for {wav_path}")

    # The references are the same for every arm and every rate, and analysing
    # them dominates the runtime, so they are analysed once and kept.
    cache_path = ref_path / f".mcep_o{args.order}_a{args.alpha}.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    cache_grew = False

    scores = {}
    missing = 0
    for gen in sorted(wav_path.glob(f"*.{args.extension}")):
        ref = ref_path / gen.name
        if not ref.exists():
            missing += 1
            continue
        if gen.stem not in cache:
            cache[gen.stem] = mel_cepstrum(
                ref, args.order, args.alpha, args.sample_rate
            )
            cache_grew = True
        b = cache[gen.stem]
        a = mel_cepstrum(gen, args.order, args.alpha, args.sample_rate)
        if len(a) < 2 or len(b) < 2:
            continue
        mean_distance, _ = dtw_mean_distance(a, b)
        scores[gen.stem] = MCD_SCALE * mean_distance

    if cache_grew:
        np.savez(cache_path, **cache)

    if missing:
        logging.warning(f"{missing} files had no reference and were skipped")
    if not scores:
        raise SystemExit("No pairs were scored")

    if args.score_path is not None:
        out = Path(args.score_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("wav_name\tMCD\n")
            for name in sorted(scores):
                f.write(f"{name}\t{scores[name]:.4f}\n")
        logging.info(f"Wrote {len(scores)} per-file scores to {args.score_path}")

    values = np.array(list(scores.values()))
    print("-" * 50)
    logging.info(f"MCD: {values.mean():.3f} dB over {len(values)} utterances")
    print("-" * 50)


if __name__ == "__main__":
    main()
