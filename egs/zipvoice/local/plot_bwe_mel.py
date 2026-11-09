#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft


def load_audio(path: str, sample_rate: int = 24000) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, source_rate // divisor)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Bandwidth-extension spectrogram comparison")
    parser.add_argument("--item", action="append", required=True, help="LABEL=PATH")
    parser.add_argument(
        "--cutoff-khz",
        type=float,
        default=4.0,
        help="Prompt band edge, drawn as a dashed line. The prompt rate is twice "
        "this: 8000 -> 4.0, 16000 -> 8.0.",
    )
    args = parser.parse_args()

    items = [item.split("=", maxsplit=1) for item in args.item]
    fig, axes = plt.subplots(len(items), 1, figsize=(14, 2.35 * len(items)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, (label, path) in zip(axes, items):
        audio = load_audio(path)
        frequencies, times, spectrum = stft(
            audio,
            fs=24000,
            window="hann",
            nperseg=1024,
            noverlap=768,
            boundary=None,
        )
        db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1.0e-7))
        image = axis.pcolormesh(times, frequencies / 1000.0, db, shading="auto", cmap="magma", vmin=-100, vmax=-20)
        axis.axhline(
            args.cutoff_khz, color="cyan", linewidth=0.9, linestyle="--", alpha=0.9
        )
        axis.set_ylim(0, 12)
        axis.set_ylabel(f"{label}\nkHz")
        axis.set_xlim(0, max(float(times[-1]), 0.1))
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(args.title)
    fig.colorbar(image, ax=axes, label="Magnitude (dBFS)", shrink=0.75)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
