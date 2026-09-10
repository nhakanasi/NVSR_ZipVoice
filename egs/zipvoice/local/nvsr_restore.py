"""Restore evaluation prompts with the released NVSR checkpoint.

This builds the cascade baseline: an off-the-shelf bandwidth extender runs on
the degraded prompt waveform, and stock ZipVoice is then conditioned on the
restored waveform. Nothing here is trained in this repository -- the weights
are the ones published with ssr_eval, downloaded on first use.

NVSR works at 44.1 kHz, so a prompt recorded at 8, 16 or 22.05 kHz is
resampled up, restored, and resampled to the 24 kHz ZipVoice expects. The
model's own post-processing step is kept: it splices the original low band
back over the restored one in the STFT domain, so the cascade is measured in
the configuration its authors recommend.

The script needs the ssr_eval source tree only for the NVSR example code,
which uses flat imports and therefore has to be on sys.path directly. The
ssr_eval package itself is not imported: it pulls in an evaluation harness
whose dependencies do not build on this platform, and the two helpers taken
from it are short enough to carry here instead.
"""

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

NVSR_SR = 44100


def find_cutoff(x: np.ndarray, threshold: float = 0.95) -> int:
    threshold = x[-1] * threshold
    for i in range(1, x.shape[0]):
        if x[-i] < threshold:
            return x.shape[0] - i
    return 0


def cutoff_index(x: np.ndarray) -> int:
    energy = np.cumsum(np.sum(np.abs(librosa.stft(x)), axis=-1))
    return find_cutoff(energy, 0.97)


def postprocess(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Splice the observed low band back over the restored signal."""
    length = out.shape[0]
    keep = cutoff_index(x)
    stft_in = librosa.stft(x)
    stft_out = librosa.stft(out)
    stft_out[:keep, ...] = stft_in[:keep, ...]
    return librosa.istft(stft_out, length=length)


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


def restore(model, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        # (batch, channel, time): wav_to_spectrogram_phase indexes a
        # channel axis, so a bare (1, T) tensor raises IndexError.
        segment = torch.tensor(x.copy(), device=device)[None, None, ...]
        sp, _, _ = model.f_helper.wav_to_spectrogram_phase(segment)
        mel = model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = model.vocoder(from_log(model(mel)["mel"]), cuda=device == "cuda")
        out = trim_to(out, segment.shape[-1]).squeeze().cpu().numpy()
    return postprocess(x, np.squeeze(out))


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

    with open(args.out_list, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
        )
        for i, row in enumerate(rows):
            x, _ = librosa.load(row[2], sr=NVSR_SR)
            y = restore(model, x, device)
            y = librosa.resample(
                y, orig_sr=NVSR_SR, target_sr=args.sampling_rate
            )
            out_path = args.prompt_dir / f"{row[0]}.wav"
            sf.write(out_path, y, args.sampling_rate)
            writer.writerow([row[0], row[1], str(out_path), row[3]])
            print(f"[{i + 1}/{len(rows)}] {out_path}", flush=True)


if __name__ == "__main__":
    main()
