"""Per-frame band energy, split into speech frames and silence frames.

The band levels reported by ``band_profile.py`` are long-term averages, and a
constant veil over the whole utterance and a band that switches on only during
fricatives can produce the same average. This tool separates them: every frame
is labelled speech or silence from its own low band, which every system
reproduces faithfully because it is the band the prompt observes, and the
invented bands are then summarised within each label.

A system that genuinely extends bandwidth raises the upper band on speech
frames and leaves the silence frames where the reference has them. A veil
raises both.

Usage:
    python band_frames.py ref=<dir> systemA=<dir> ...
"""

import glob
import os
import sys

import numpy as np
import soundfile as sf

# Reported bands, in Hz. The first is the reference band every system observes
# through the prompt; it is what frames are labelled with and what the other
# bands are expressed relative to.
LOW = (100, 2000)
BANDS = [(4000, 6000), (6000, 8000), (8000, 12000)]

WIN = 1024
HOP = 256

# A frame counts as speech when its low band is within this many dB of the
# utterance's own 95th percentile. The reference decides the labels for its own
# file; every system is labelled from its own low band, which is legitimate here
# only because the low band is the part all of them get right.
SPEECH_DB = 35.0


def frame_bands(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < WIN:
        return None
    win = np.hanning(WIN)
    idx = range(0, len(x) - WIN + 1, HOP)
    spec = np.stack([np.abs(np.fft.rfft(x[i : i + WIN] * win)) ** 2 for i in idx])
    freq = np.fft.rfftfreq(WIN, 1 / sr)

    def band(lo, hi):
        sel = (freq >= lo) & (freq < hi)
        return spec[:, sel].sum(axis=1) + 1e-20

    low = band(*LOW)
    return low, np.stack([band(lo, hi) for lo, hi in BANDS])


def summarise(directory):
    speech, silence = [], []
    files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    for path in files:
        got = frame_bands(path)
        if got is None:
            continue
        low, bands = got
        low_db = 10 * np.log10(low)
        is_speech = low_db > np.percentile(low_db, 95) - SPEECH_DB
        # Each band in dB relative to this utterance's mean low band during
        # speech, so systems with different overall gain stay comparable.
        ref = 10 * np.log10(low[is_speech].mean()) if is_speech.any() else 0.0
        rel = 10 * np.log10(bands) - ref
        if is_speech.any():
            speech.append(rel[:, is_speech])
        if (~is_speech).any():
            silence.append(rel[:, ~is_speech])
    if not speech:
        return None
    sp = np.concatenate(speech, axis=1)
    si = np.concatenate(silence, axis=1) if silence else np.zeros((len(BANDS), 1))
    return sp, si, len(files)


def main(args):
    head = "".join(f"{a // 1000}-{b // 1000}k".rjust(21) for a, b in BANDS)
    print(f"{'system':22s}{head}")
    print(f"{'':22s}" + "".join("   speech  silence  swing" for _ in BANDS))
    for arg in args:
        directory, _, label = arg.partition("=")
        if not label:
            directory, label = arg, os.path.basename(arg.rstrip("/\\"))
        got = summarise(directory)
        if got is None:
            print(f"{label:22s}   no .wav files under {directory}")
            continue
        sp, si, n = got
        cells = ""
        for b in range(len(BANDS)):
            swing = np.percentile(sp[b], 90) - np.percentile(sp[b], 10)
            cells += f"{sp[b].mean():9.1f}{si[b].mean():9.1f}{swing:7.1f}"
        print(f"{label:22s}{cells}   n={n}")


if __name__ == "__main__":
    main(sys.argv[1:])
