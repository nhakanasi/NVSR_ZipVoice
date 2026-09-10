#!/usr/bin/env python3
# Copyright    2025    Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
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

"""Train the mel-domain F0 and voicing estimator.

AnyBand conditions on an F0 contour and a voicing flag extracted from the
band-limited input waveform with pYIN. A text-to-speech model has no waveform
over the span it is generating, so this trains a small network to read the same
two quantities off a mel instead. It is trained once, here, against pYIN targets,
and thereafter frozen: inside the flow-matching model it is a fixed feature
extractor, not something that adapts.

Training it on the audio directly, rather than folding it into the main run,
keeps the main run's dataloader untouched -- it carries precomputed mels and no
waveform -- and means the estimator can be trained on a few tens of hours, which
is ample for a quantity as low-dimensional as F0.

Usage:

python3 -m zipvoice.bin.train_f0_estimator \
    --manifest data/fbank/libritts_cuts_train-clean-100.jsonl.gz \
    --num-cuts 4000 \
    --num-epochs 20 \
    --exp-dir exp/f0_estimator
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lhotse import CutSet
from torch.utils.data import DataLoader, Dataset

from zipvoice.models.modules.anyband import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    MelF0Estimator,
    normalize_f0,
)
from zipvoice.utils.feature import VocosFbank


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="A lhotse cut manifest whose cuts still have their recordings, so "
        "that pYIN has a waveform to work from.",
    )
    parser.add_argument(
        "--num-cuts",
        type=int,
        default=4000,
        help="How many cuts to use. F0 estimation is a small problem and a few "
        "thousand utterances saturate it; the cost here is dominated by pYIN.",
    )
    parser.add_argument(
        "--num-valid-cuts",
        type=int,
        default=200,
        help="Cuts held out to report accuracy on.",
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--channels",
        type=int,
        default=192,
        help="Width of the estimator; must match the model config it is loaded "
        "into.",
    )
    parser.add_argument("--sampling-rate", type=int, default=24000)
    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="Scale the mel is multiplied by, matching training of the "
        "flow-matching model so the estimator sees the units it will be given.",
    )
    parser.add_argument(
        "--band-limit-prob",
        type=float,
        default=0.5,
        help="Probability of low-passing the mel before the estimator sees it. "
        "At inference it reads an endpoint estimate formed from a band-limited "
        "prompt, so it has to be robust to a missing high band. F0 itself lies "
        "below 1100 Hz and survives any realistic cutoff, which is why this is "
        "a robustness measure rather than a limitation.",
    )
    parser.add_argument(
        "--voiced-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the voicing cross-entropy against the F0 regression.",
    )
    parser.add_argument("--exp-dir", type=str, default="exp/f0_estimator")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


class F0Dataset(Dataset):
    """Mels paired with a pYIN F0 contour and voicing flag.

    pYIN is run at the mel's own hop so the two are frame-aligned by
    construction; any residual off-by-one from the framing is trimmed rather
    than interpolated.
    """

    def __init__(
        self,
        cuts: CutSet,
        sampling_rate: int,
        feat_scale: float,
        cache_dir: Optional[Path] = None,
    ):
        self.cuts = list(cuts)
        self.sampling_rate = sampling_rate
        self.feat_scale = feat_scale
        self.fbank = VocosFbank()
        # pYIN dominates the cost of an epoch and its targets never change, so
        # the first epoch writes them out and every later one reads them back.
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.cuts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, ...]]:
        import librosa

        cut = self.cuts[idx]
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / f"{cut.id}.npz"
            if cache.is_file():
                try:
                    d = np.load(cache)
                    return (
                        torch.from_numpy(d["mel"]),
                        torch.from_numpy(d["f0"]),
                        torch.from_numpy(d["voiced"]),
                    )
                except Exception:  # a truncated file from an interrupted run
                    cache.unlink(missing_ok=True)
        try:
            audio = cut.resample(self.sampling_rate).load_audio()
        except Exception as exc:  # a missing or unreadable recording
            logging.warning(f"Skipping {cut.id}: {exc}")
            return None
        wav = torch.from_numpy(audio).float().squeeze(0)

        mel = self.fbank.extract(wav, sampling_rate=self.sampling_rate)
        mel = mel * self.feat_scale

        f0, voiced_flag, _ = librosa.pyin(
            wav.numpy(),
            fmin=F0_MIN_HZ,
            fmax=F0_MAX_HZ,
            sr=self.sampling_rate,
            frame_length=1024,
            hop_length=256,
            center=True,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        f0 = np.where(voiced_flag, f0, 0.0)

        num_frames = min(mel.shape[0], f0.shape[0])
        mel = mel[:num_frames]
        f0_t = torch.from_numpy(f0[:num_frames]).float()
        voiced = torch.from_numpy(voiced_flag[:num_frames].astype("float32"))
        f0_n = normalize_f0(f0_t)
        if cache is not None:
            np.savez(
                cache,
                mel=mel.numpy(),
                f0=f0_n.numpy(),
                voiced=voiced.numpy(),
            )
        return mel, f0_n, voiced


def collate(batch: List[Optional[Tuple[torch.Tensor, ...]]]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    lengths = torch.tensor([b[0].shape[0] for b in batch])
    max_len = int(lengths.max())
    num_bins = batch[0][0].shape[1]
    mel = torch.zeros(len(batch), max_len, num_bins)
    f0 = torch.zeros(len(batch), max_len)
    voiced = torch.zeros(len(batch), max_len)
    for i, (m, f, v) in enumerate(batch):
        mel[i, : m.shape[0]] = m
        f0[i, : f.shape[0]] = f
        voiced[i, : v.shape[0]] = v
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return mel, f0, voiced, mask


def band_limit_mel(mel: torch.Tensor, prob: float) -> torch.Tensor:
    """Zero the top of a random fraction of the batch's mels, above 4-12 kHz.

    Deliberately cruder than the flow-matching model's band-limiting: the point
    is only that the estimator must not come to depend on the high band.
    """
    if prob <= 0:
        return mel
    out = mel.clone()
    num_bins = mel.shape[-1]
    apply = torch.rand(mel.shape[0]) < prob
    for i in torch.nonzero(apply).flatten().tolist():
        cutoff = int(torch.randint(int(num_bins * 0.55), num_bins, (1,)))
        out[i, :, cutoff:] = mel[i].min()
    return out


def compute_loss(
    model: MelF0Estimator,
    mel: torch.Tensor,
    f0: torch.Tensor,
    voiced: torch.Tensor,
    mask: torch.Tensor,
    voiced_loss_weight: float,
) -> Tuple[torch.Tensor, dict]:
    f0_pred, voiced_logit = model(mel)

    # The F0 regression is scored on voiced frames only. Unvoiced frames have no
    # F0 to predict -- their target is the sentinel 0 -- so including them would
    # train the network to output that sentinel, which is the voicing head's job.
    voiced_mask = mask & (voiced > 0.5)
    denom = voiced_mask.sum().clamp(min=1)
    f0_loss = (F.l1_loss(f0_pred, f0, reduction="none") * voiced_mask).sum() / denom

    voiced_loss = (
        F.binary_cross_entropy_with_logits(voiced_logit, voiced, reduction="none")
        * mask
    ).sum() / mask.sum().clamp(min=1)

    loss = f0_loss + voiced_loss_weight * voiced_loss
    correct = ((voiced_logit > 0).float() == voiced) & mask
    stats = {
        "f0_l1": f0_loss.item(),
        "voiced_bce": voiced_loss.item(),
        "voiced_acc": (correct.sum() / mask.sum().clamp(min=1)).item(),
    }
    return loss, stats


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    cuts = CutSet.from_file(args.manifest)
    cuts = cuts.subset(first=args.num_cuts + args.num_valid_cuts)
    all_cuts = list(cuts)
    valid_cuts = CutSet.from_cuts(all_cuts[: args.num_valid_cuts])
    train_cuts = CutSet.from_cuts(all_cuts[args.num_valid_cuts :])
    logging.info(f"{len(train_cuts)} training cuts, {len(valid_cuts)} validation")

    cache_root = exp_dir / "pyin_cache"
    make = lambda c: F0Dataset(  # noqa: E731
        c, args.sampling_rate, args.feat_scale, cache_root
    )
    train_dl = DataLoader(
        make(train_cuts),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    valid_dl = DataLoader(
        make(valid_cuts),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model = MelF0Estimator(num_bins=100, channels=args.channels).to(device)
    logging.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for step, batch in enumerate(train_dl):
            if batch is None:
                continue
            mel, f0, voiced, mask = (x.to(device) for x in batch)
            loss, stats = compute_loss(
                model,
                band_limit_mel(mel, args.band_limit_prob),
                f0,
                voiced,
                mask,
                args.voiced_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step % 50 == 0:
                logging.info(f"epoch {epoch} step {step} {stats}")

        model.eval()
        totals, count = {}, 0
        with torch.no_grad():
            for batch in valid_dl:
                if batch is None:
                    continue
                mel, f0, voiced, mask = (x.to(device) for x in batch)
                _, stats = compute_loss(
                    model, mel, f0, voiced, mask, args.voiced_loss_weight
                )
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0.0) + v
                count += 1
        stats = {k: v / max(count, 1) for k, v in totals.items()}
        logging.info(f"epoch {epoch} validation {stats}")

        torch.save({"model": model.state_dict()}, exp_dir / "last.pt")
        if stats.get("voiced_acc", 0.0) > best_acc:
            best_acc = stats["voiced_acc"]
            torch.save({"model": model.state_dict()}, exp_dir / "best.pt")
            logging.info(f"New best voicing accuracy {best_acc:.4f}, saved best.pt")

    logging.info(f"Done. Best voicing accuracy {best_acc:.4f}")


if __name__ == "__main__":
    main()
