#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
Calculate DNSMOS P.835 (SIG / BAK / OVRL) and DNSMOS P.808 scores.

UTMOS predicts a single overall naturalness opinion score. DNSMOS P.835 splits
the judgement into three axes -- speech distortion, background intrusiveness and
overall quality -- which separates "the voice itself sounds clean" from "there is
audible junk around it". That distinction is what a bandwidth extender is
expected to move.

Ported from the reference implementation in microsoft/DNS-Challenge
(DNSMOS/dnsmos_local.py). Two deviations, both deliberate:

  - Audio is loaded through `zipvoice.eval.utils.load_waveform`, so resampling
    matches every other evaluator in this package. The reference calls
    `librosa.resample` positionally, which no longer works on librosa >= 0.10.
  - Per-file scores can be written out with --score-path, so conditions can be
    compared with a paired test instead of only by their means.

Both ONNX models come from the DNS-Challenge repository:

    curl -sL -o sig_bak_ovr.onnx \
      https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
    curl -sL -o model_v8.onnx \
      https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx

and belong in `<model-dir>/mos/`.

Note on scope: DNSMOS runs at 16 kHz, so it only ever sees content below 8 kHz.
For an 8 kHz prompt the extender predicts 4-12 kHz and DNSMOS observes the
4-8 kHz half of that. For a 16 kHz or higher prompt everything the extender
predicts is above 8 kHz and therefore invisible here; those rows measure the
rest of the synthesis, not the extender's output.
"""
import argparse
import logging
import os
from typing import Dict, List

import librosa
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from zipvoice.eval.utils import load_waveform

SAMPLING_RATE = 16000
# Fixed input length of the P.835 model, in seconds.
INPUT_LENGTH = 9.01

METRICS = ["SIG", "BAK", "OVRL", "P808_MOS"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate DNSMOS P.835 and P.808 scores."
    )
    parser.add_argument(
        "--wav-path",
        type=str,
        required=True,
        help="Path to the directory containing evaluated speech files.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="""Local path of our evaluation model repository. Will use
        'mos/sig_bak_ovr.onnx' and 'mos/model_v8.onnx' in this script; see the
        module docstring for where to fetch them.
        """,
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )
    parser.add_argument(
        "--score-path",
        type=str,
        default=None,
        help="""Optional TSV to write per-file scores to. Needed for paired
        comparisons between conditions; the printed summary is means only.
        """,
    )
    parser.add_argument(
        "--personalized",
        action="store_true",
        help="""Use the personalized-MOS polynomial mapping, for models trained
        to suppress interfering speakers. Off by default, which is correct for
        single-speaker TTS output.
        """,
    )
    return parser


class DNSMOSScore:
    """Predicting DNSMOS P.835 and P.808 scores for each audio clip."""

    def __init__(
        self,
        primary_model_path: str,
        p808_model_path: str,
        personalized: bool = False,
    ):
        self.sample_rate = SAMPLING_RATE
        self.personalized = personalized
        # Both models are small and run on a 9 s clip at a time; CPU is quick
        # enough and avoids fighting the TTS job for the GPU.
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(
        self,
        audio: np.ndarray,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        to_db: bool = True,
    ) -> np.ndarray:
        """Mel spectrogram in the exact form the P.808 model expects."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def polyfit(self, sig: float, bak: float, ovr: float) -> List[float]:
        """Map raw model outputs onto the calibrated P.835 opinion scale."""
        if self.personalized:
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
        else:
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        return [float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))]

    def score_file(self, wav_path: str) -> Dict[str, float]:
        """Score one file, averaging over 9.01 s windows hopped by 1 s."""
        audio = load_waveform(wav_path, self.sample_rate, return_numpy=True)

        len_samples = int(INPUT_LENGTH * self.sample_rate)
        # The model has a fixed input length, so a clip shorter than that is
        # tiled rather than zero-padded: silence would be scored as background.
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / self.sample_rate) - INPUT_LENGTH) + 1
        seg_scores: List[List[float]] = []
        for idx in range(num_hops):
            start = idx * self.sample_rate
            seg = audio[start : start + len_samples]
            if len(seg) < len_samples:
                continue

            features = np.array(seg).astype("float32")[np.newaxis, :]
            sig_raw, bak_raw, ovr_raw = self.onnx_sess.run(
                None, {"input_1": features}
            )[0][0]

            p808_features = np.array(self.audio_melspec(seg[:-160])).astype("float32")[
                np.newaxis, :, :
            ]
            p808 = self.p808_onnx_sess.run(None, {"input_1": p808_features})[0][0][0]

            seg_scores.append(self.polyfit(sig_raw, bak_raw, ovr_raw) + [float(p808)])

        if not seg_scores:
            raise ValueError(f"No scoreable window in {wav_path}")

        means = np.mean(np.array(seg_scores), axis=0)
        return dict(zip(METRICS, means.tolist()))

    def score_dir(self, dir_path: str, extension: str) -> Dict[str, Dict[str, float]]:
        """Score every file in a directory, keyed by file name."""
        logging.info(f"Calculating DNSMOS scores for {dir_path}")

        wav_files = sorted(
            f for f in os.listdir(dir_path) if f.lower().endswith(extension)
        )
        if not wav_files:
            raise ValueError(f"No audio files found in {dir_path}")

        return {
            f: self.score_file(os.path.join(dir_path, f))
            for f in tqdm(wav_files, desc="Scoring audio files")
        }


if __name__ == "__main__":

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.wav_path):
        logging.error(f"Invalid directory: {args.wav_path}")
        exit(1)

    primary_model_path = os.path.join(args.model_dir, "mos/sig_bak_ovr.onnx")
    p808_model_path = os.path.join(args.model_dir, "mos/model_v8.onnx")
    for path in (primary_model_path, p808_model_path):
        if not os.path.exists(path):
            logging.error(
                f"Missing {path}. Download the DNSMOS ONNX models from "
                "https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS"
                " into the 'mos' subdirectory of --model-dir."
            )
            exit(1)

    evaluator = DNSMOSScore(primary_model_path, p808_model_path, args.personalized)
    scores = evaluator.score_dir(args.wav_path, args.extension)

    if args.score_path is not None:
        with open(args.score_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(["wav_name"] + METRICS) + "\n")
            for name, row in scores.items():
                values = [f"{row[m]:.4f}" for m in METRICS]
                f.write("\t".join([os.path.splitext(name)[0]] + values) + "\n")
        logging.info(f"Wrote {len(scores)} per-file scores to {args.score_path}")

    print("-" * 50)
    for metric in METRICS:
        mean = float(np.mean([row[metric] for row in scores.values()]))
        logging.info(f"DNSMOS {metric}: {mean:.3f}")
    print("-" * 50)
