"""Restore evaluation prompts with the released NVSR checkpoint.

This builds the cascade baseline: an off-the-shelf bandwidth extender runs on
the degraded prompt waveform, and stock ZipVoice is then conditioned on the
restored waveform. Nothing here is trained in this repository -- the weights
are the ones published with ssr_eval, downloaded on first use.

NVSR works at 44.1 kHz, so a prompt recorded at 8, 16 or 22.05 kHz is
resampled up, restored, and resampled to the 24 kHz ZipVoice expects. The
observed band is then put back over the restored one by :func:`lfr`, which is
also the low-frequency replacement lfr.py applies to the bandmask model's own
prompt fill, so both extensions are spliced by one definition.

NVSR's released post-processing spliced at the bin where the input's
cumulative energy over frequency reaches 97%. That is an energy quantile, not
the band edge: on the 8 kHz evaluation prompts it sits near 3.2 kHz, and on the
16 kHz ones anywhere from 4.5 to 7.3 kHz, so NVSR's resynthesis replaced the
top of the band the prompt already carried (the bwe_eval nvsr* prompts and the
Control 2b restorations up to 2026-10-03 were made that way). :func:`lfr`
splices just below the detected edge instead.

The script needs the ssr_eval source tree only for the NVSR example code,
which uses flat imports and therefore has to be on sys.path directly. The
ssr_eval package itself is not imported: it pulls in an evaluation harness
whose dependencies do not build on this platform, and the two helpers taken
from it are short enough to carry here instead.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch

from zipvoice.models.modules.bwe import (
    BAND_EDGE_LEAKAGE_BINS,
    estimate_band_edge,
    mel_bin_centers,
)
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav

# torchlibrosa, which NVSR's STFT front end uses, calls
# librosa.util.pad_center(window, n_fft) positionally; librosa made that
# argument keyword-only in 0.10. The same adaptation is in audiosr_restore.py,
# for the same reason: pinning librosa back has no wheel for this Python.
_pad_center = librosa.util.pad_center


def _pad_center_compat(data, size=None, *args, **kwargs):
    if size is not None:
        kwargs["size"] = size
    return _pad_center(data, *args, **kwargs)


librosa.util.pad_center = _pad_center_compat

NVSR_SR = 44100
# NVSR's own STFT: librosa's defaults (Hann window) at 44.1 kHz, 21.5 Hz a bin.
N_FFT, HOP = 2048, 512
# The splice is centred at this fraction of the band edge, below the
# resampler's roll-off, and crossfades linearly over this many bins.
SPLICE_FRACTION = 0.9
CROSSFADE_BINS = 8


def band_cutoff(path: Path) -> Optional[float]:
    """The band edge of a prompt file, in Hz.

    Uses the estimator the bandmask model masks its prompt with, on the prompt
    loaded as inference loads it, so the cascade and the model take the same
    edge. The loader matters: the estimator's leakage correction is calibrated
    on torchaudio's resampler, whose roll-off leaks; a prompt brought to 24 kHz
    by a sharper one reads about 5 bins lower. Returns None for a prompt whose
    band reaches the top of the estimator's range: there is nothing to extend.
    """
    wav, _ = load_prompt_wav(str(path), 24000)
    features = VocosFbank().extract(wav[0], 24000)
    edge = float(estimate_band_edge(features[None], torch.tensor([len(features)]))[0])
    n_mels = features.shape[-1]
    if edge >= n_mels - 1:
        return None
    centers = mel_bin_centers(n_mels).numpy()
    return float(np.interp(edge - BAND_EDGE_LEAKAGE_BINS, np.arange(n_mels), centers))


def lfr(low: np.ndarray, full: np.ndarray, cutoff: Optional[float]) -> np.ndarray:
    """Low-frequency replacement: the observed band from ``low``, the rest from
    ``full``, both 44.1 kHz waveforms.

    The STFT (magnitude and phase) takes ``low`` below the splice and ``full``
    above it, with a linear crossfade of :data:`CROSSFADE_BINS` bins centred at
    :data:`SPLICE_FRACTION` of ``cutoff`` (Hz). A ``cutoff`` of None returns
    ``low``.
    """
    n = min(len(low), len(full))
    if cutoff is None:
        return low[:n]
    centre = SPLICE_FRACTION * cutoff * N_FFT / NVSR_SR
    weight = np.clip((centre + CROSSFADE_BINS / 2 - np.arange(N_FFT // 2 + 1)) / CROSSFADE_BINS, 0, 1)
    spec_low = librosa.stft(low[:n], n_fft=N_FFT, hop_length=HOP)
    spec_full = librosa.stft(full[:n], n_fft=N_FFT, hop_length=HOP)
    spec = weight[:, None] * spec_low + (1 - weight[:, None]) * spec_full
    return librosa.istft(spec, hop_length=HOP, length=n)


def from_log(x: torch.Tensor) -> torch.Tensor:
    return 10 ** torch.clip(x, min=-np.inf, max=5)


def trim_to(est: torch.Tensor, length: int) -> torch.Tensor:
    if est.shape[-1] <= length:
        return est
    diff = est.shape[-1] - length
    return est[..., diff // 2: diff // 2 + length]


def load_nvsr(nvsr_dir: Path, device: str):
    sys.path.insert(0, str(nvsr_dir))
    from nvsr_unet import NVSR  # noqa: E402  (path set immediately above)

    ckpt = (
        Path.home()
        / ".cache/ssr_eval/NVSR/epoch=11-step=22499-val_l=0.27.pth"
    )
    if not ckpt.is_file():
        import urllib.request

        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the released NVSR weights to {ckpt}", flush=True)
        urllib.request.urlretrieve(
            "https://zenodo.org/record/6370601/files/"
            "epoch%3D11-step%3D22499-val_l%3D0.27.pth?download=1",
            ckpt,
        )

    model = NVSR(channels=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval().to(device)


def extend(model, x: np.ndarray, device: str) -> np.ndarray:
    """NVSR's restoration of a 44.1 kHz waveform, before any splice."""
    with torch.no_grad():
        # (batch, channel, time): wav_to_spectrogram_phase indexes a
        # channel axis, so a bare (1, T) tensor raises IndexError.
        segment = torch.tensor(x.copy(), device=device)[None, None, ...]
        sp, _, _ = model.f_helper.wav_to_spectrogram_phase(segment)
        mel = model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = model.vocoder(from_log(model(mel)["mel"]), cuda=device == "cuda")
        out = trim_to(out, segment.shape[-1]).squeeze().cpu().numpy()
    return np.squeeze(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument(
        "--nvsr-dir",
        type=Path,
        default=Path("C:/zipvoice_data/ssr_eval/examples/NVSR"),
    )
    parser.add_argument("--sampling-rate", type=int, default=24000)
    parser.add_argument(
        "--timing-path",
        type=Path,
        default=None,
        help="Optional TSV of per-prompt wall time, for the cascade's "
        "real-time factor. One row per prompt actually restored.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_nvsr(args.nvsr_dir, device)
    args.prompt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_list, encoding="utf-8") as handle:
        # QUOTE_NONE: the transcripts contain bare double quotes, which
        # the default dialect reads as field quoting -- it silently
        # merges rows (76 -> 70 on test_sr8000.tsv) and strips the marks.
        rows = list(
            csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        )

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
            # Named by the prompt, not by the row: the same prompt serves
            # several target sentences, and restoring it once per row would
            # both waste the pass and inflate the measured real-time factor.
            # An existing file also makes an interrupted run resumable.
            out_path = args.prompt_dir / f"{source.stem}.wav"
            if source not in restored:
                if out_path.is_file():
                    restored[source] = out_path
                else:
                    x, _ = librosa.load(source, sr=NVSR_SR)
                    start = time.time()
                    y = lfr(x, extend(model, x, device), band_cutoff(source))
                    elapsed = time.time() - start
                    y = librosa.resample(
                        y, orig_sr=NVSR_SR, target_sr=args.sampling_rate
                    )
                    # Float, not soundfile's default 16-bit PCM: the
                    # quantisation floor alone lifts the observed band's LSD
                    # against the clean prompt from 0.053 to 0.068 (8 kHz).
                    sf.write(out_path, y, args.sampling_rate, subtype="FLOAT")
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
