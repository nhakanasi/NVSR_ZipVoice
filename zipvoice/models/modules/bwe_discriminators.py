# Copyright    2026
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
The three-view discriminator set of AnyBand (arXiv 2608.00572), adapted to
ZipVoice's 100-bin 24 kHz VocosFbank mel:

  * ``SpecDiscriminator``  -- multi-scale over time, checks spectral realism
  * ``CrossBandDiscriminator`` -- checks that the generated high band follows the
    temporal dynamics of the observed low band (envelope coherence)
  * ``HarmonicDiscriminator`` -- checks harmonic consistency along F0 tracks

Deviations from the paper are noted at each class.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipvoice.models.modules.bwe import hz_to_mel_bin

FeatureMaps = List[torch.Tensor]


def _conv_stack(in_channels: int, channels: List[int]) -> nn.ModuleList:
    """A LeakyReLU 2D conv stack that halves both axes at every layer."""
    layers = nn.ModuleList()
    prev = in_channels
    for ch in channels:
        layers.append(
            nn.utils.weight_norm(nn.Conv2d(prev, ch, 3, stride=2, padding=1))
        )
        prev = ch
    layers.append(nn.utils.weight_norm(nn.Conv2d(prev, 1, 3, padding=1)))
    return layers


def _run_stack(
    layers: nn.ModuleList, x: torch.Tensor
) -> Tuple[torch.Tensor, FeatureMaps]:
    features: FeatureMaps = []
    for layer in layers[:-1]:
        x = F.leaky_relu(layer(x), 0.1)
        features.append(x)
    return layers[-1](x), features


class SpecDiscriminator(nn.Module):
    """
    Multi-scale spectral discriminator: independent 2D conv branches on the mel
    average-pooled along time by 1, 1/2 and 1/4. Frequency resolution is
    preserved. Channels 32/64/128/256, per the paper.
    """

    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        channels: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList(
            [_conv_stack(1, channels) for _ in scales]
        )

    def forward(self, mel: torch.Tensor) -> Tuple[List[torch.Tensor], FeatureMaps]:
        """
        Args:
            mel: log-mel, shape ``(B, T, F)``.

        Returns:
            ``(logits, features)``: one logit map per scale, and the concatenated
            intermediate feature maps of all branches.
        """
        x = mel.unsqueeze(1)  # (B, 1, T, F)
        logits, features = [], []
        for scale, branch in zip(self.scales, self.branches):
            h = x if scale == 1 else F.avg_pool2d(x, (scale, 1))
            logit, feats = _run_stack(branch, h)
            logits.append(logit)
            features.extend(feats)
        return logits, features


class CrossBandDiscriminator(nn.Module):
    """
    Cross-band envelope coherence discriminator.

    The mel is split into contiguous subbands; each subband's temporal envelope
    is mean-pooled over its bins and standardized over time, then multiplied by
    the observed-fraction-weighted low-band reference envelope and locally
    averaged over a window. The resulting coherence map, stacked with the
    observed-fraction map, is the discriminator input.

    Deviation: the paper uses 16 subbands of 8 bins over a 128-bin mel. 100 bins
    is not divisible by 16, so this uses 20 subbands of 5 bins.
    """

    def __init__(
        self,
        n_mels: int = 100,
        num_subbands: int = 20,
        window: int = 32,
        channels: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        assert n_mels % num_subbands == 0, (n_mels, num_subbands)
        self.n_mels = n_mels
        self.num_subbands = num_subbands
        self.bins_per_band = n_mels // num_subbands
        self.window = window
        self.layers = _conv_stack(2, channels)

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        """Zero-mean unit-variance along the time axis (last dim)."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-5)
        return (x - mean) / std

    def forward(
        self, mel: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], FeatureMaps]:
        """
        Args:
            mel: log-mel, shape ``(B, T, F)``.
            mask: band mask, shape ``(B, 1, F)``, 1.0 where the band is missing.

        Returns:
            ``(logits, features)``.
        """
        b, t, _ = mel.shape
        # (B, num_subbands, T)
        envelopes = (
            mel.transpose(1, 2)
            .reshape(b, self.num_subbands, self.bins_per_band, t)
            .mean(dim=2)
        )
        envelopes = self._standardize(envelopes)

        observed = 1.0 - mask  # (B, 1, F)
        # Observed fraction of each subband, (B, num_subbands)
        rho = observed.reshape(b, self.num_subbands, self.bins_per_band).mean(dim=2)

        # Low-band reference envelope: mel averaged over observed bins.
        weight = observed.transpose(1, 2)  # (B, F, 1)
        denom = weight.sum(dim=1, keepdim=True).clamp(min=1e-5)  # (B, 1, 1)
        reference = (mel.transpose(1, 2) * weight).sum(dim=1, keepdim=True) / denom
        reference = self._standardize(reference)  # (B, 1, T)

        coherence = envelopes * reference * rho.unsqueeze(-1)
        # Local average over `window` frames.
        pad = self.window // 2
        coherence = F.avg_pool1d(
            F.pad(coherence, (pad, self.window - 1 - pad), mode="replicate"),
            kernel_size=self.window,
            stride=1,
        )

        x = torch.stack(
            [coherence, rho.unsqueeze(-1).expand(b, self.num_subbands, t)], dim=1
        )  # (B, 2, num_subbands, T)
        logit, features = _run_stack(self.layers, x)
        return [logit], features


class HarmonicDiscriminator(nn.Module):
    """
    Harmonic-consistency discriminator: samples the mel along the first
    ``num_harmonics`` F0-aligned harmonics and judges the resulting grid,
    stacked with an observed-harmonic mask and a voiced/unvoiced map.

    Caveat: with 100 mel bins over 24 kHz, F0 (80-400 Hz) occupies roughly bins
    2-14, so the F0 track itself is coarse. Its adversarial loss is logged
    separately so its contribution can be judged on evidence.
    """

    def __init__(
        self,
        num_harmonics: int = 12,
        channels: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.layers = _conv_stack(3, channels)

    def forward(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
        voiced: torch.Tensor,
        cutoff_hz: torch.Tensor,
        centers: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], FeatureMaps]:
        """
        Args:
            mel: log-mel, shape ``(B, T, F)``.
            f0: fundamental frequency in Hz per frame, shape ``(B, T)``.
            voiced: voiced/unvoiced flag per frame, shape ``(B, T)``, in [0, 1].
            cutoff_hz: cutoff frequency per utterance, shape ``(B,)``.
            centers: mel filter center frequencies, shape ``(F,)``.

        Returns:
            ``(logits, features)``.
        """
        b, t, n_mels = mel.shape
        n_h = self.num_harmonics
        orders = torch.arange(1, n_h + 1, device=mel.device, dtype=mel.dtype)
        harmonic_hz = f0.unsqueeze(1) * orders.view(1, n_h, 1)  # (B, N_h, T)

        bins = hz_to_mel_bin(harmonic_hz, centers)  # (B, N_h, T)
        x_norm = 2.0 * bins / max(n_mels - 1, 1) - 1.0
        y_norm = (
            2.0 * torch.arange(t, device=mel.device, dtype=mel.dtype) / max(t - 1, 1)
            - 1.0
        )
        grid = torch.stack(
            [x_norm, y_norm.view(1, 1, t).expand(b, n_h, t)], dim=-1
        )  # (B, N_h, T, 2)

        harmonics = F.grid_sample(
            mel.unsqueeze(1),  # (B, 1, T, F): H is time, W is frequency
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, 1, N_h, T)

        observed = (harmonic_hz <= cutoff_hz.view(b, 1, 1)).to(mel.dtype)
        x = torch.cat(
            [
                harmonics,
                observed.unsqueeze(1),
                voiced.view(b, 1, 1, t).expand(b, 1, n_h, t),
            ],
            dim=1,
        )  # (B, 3, N_h, T)
        logit, features = _run_stack(self.layers, x)
        return [logit], features


@torch.no_grad()
def f0_from_mel(
    mel: torch.Tensor,
    centers: torch.Tensor,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
    num_candidates: int = 64,
    num_harmonics: int = 8,
    voiced_threshold: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate F0 per frame directly from the mel by harmonic-comb template
    matching. No gradient flows through F0 -- it only positions the sampling
    grid of :class:`HarmonicDiscriminator` -- so a coarse estimate suffices and
    the waveform is not needed.

    The mel is standardized per frame before scoring, which makes the score and
    hence ``voiced_threshold`` independent of ``feat_scale``.

    Args:
        mel: log-mel, shape ``(B, T, F)``.
        centers: mel filter center frequencies, shape ``(F,)``.

    Returns:
        ``(f0, voiced)``, both of shape ``(B, T)``; ``f0`` in Hz and ``voiced``
        a 0/1 float mask.
    """
    b, t, n_mels = mel.shape
    dtype = mel.dtype

    mean = mel.mean(dim=-1, keepdim=True)
    std = mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
    normed = (mel - mean) / std  # (B, T, F)

    candidates = torch.logspace(
        math.log10(f0_min), math.log10(f0_max), num_candidates, device=mel.device
    ).to(dtype)
    orders = torch.arange(1, num_harmonics + 1, device=mel.device, dtype=dtype)
    comb_hz = candidates.view(num_candidates, 1) * orders.view(1, num_harmonics)
    comb_bins = hz_to_mel_bin(comb_hz, centers).round().long().clamp(0, n_mels - 1)

    # (B, T, num_candidates * num_harmonics) -> mean over harmonics
    gathered = normed[..., comb_bins.reshape(-1)]
    scores = gathered.reshape(b, t, num_candidates, num_harmonics).mean(dim=-1)

    best, best_idx = scores.max(dim=-1)
    f0 = candidates[best_idx]
    voiced = (best > voiced_threshold).to(dtype)
    return f0 * voiced, voiced


@torch.no_grad()
def f0_from_audio(
    audio: torch.Tensor,
    num_frames: int,
    sampling_rate: int = 24000,
    hop_length: int = 256,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate F0 from the waveform with ``torchaudio.functional.detect_pitch_frequency``
    and resample the track to the mel frame rate.

    More accurate than :func:`f0_from_mel`, at the cost of loading audio in the
    dataloader.

    Args:
        audio: waveform, shape ``(B, num_samples)``.
        num_frames: number of mel frames to interpolate the track onto.

    Returns:
        ``(f0, voiced)``, both of shape ``(B, num_frames)``.
    """
    import torchaudio

    track = torchaudio.functional.detect_pitch_frequency(
        audio,
        sample_rate=sampling_rate,
        frame_time=hop_length / sampling_rate,
        freq_low=int(f0_min),
        freq_high=int(f0_max),
    )  # (B, n)
    f0 = F.interpolate(
        track.unsqueeze(1), size=num_frames, mode="linear", align_corners=False
    ).squeeze(1)
    voiced = ((f0 >= f0_min) & (f0 <= f0_max)).to(audio.dtype)
    return f0 * voiced, voiced


def hinge_d_loss(
    real_logits: List[torch.Tensor], fake_logits: List[torch.Tensor]
) -> torch.Tensor:
    """Hinge discriminator loss, averaged over the logit maps."""
    loss = sum(F.relu(1.0 - r).mean() for r in real_logits)
    loss = loss + sum(F.relu(1.0 + f).mean() for f in fake_logits)
    return loss / max(len(real_logits) + len(fake_logits), 1)


def hinge_g_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge generator loss ``-E[D(fake)]``, averaged over the logit maps."""
    return -sum(f.mean() for f in fake_logits) / max(len(fake_logits), 1)


def feature_matching_loss(
    real_features: FeatureMaps, fake_features: FeatureMaps
) -> torch.Tensor:
    """L1 between the discriminator's intermediate feature maps."""
    if not real_features:
        return torch.zeros((), device=fake_features[0].device)
    loss = sum(
        F.l1_loss(f, r.detach()) for r, f in zip(real_features, fake_features)
    )
    return loss / len(real_features)


class BweDiscriminators(nn.Module):
    """
    The three discriminators together, exposing one forward that returns the
    logits and features of every view, so the training loop stays flat.
    """

    def __init__(
        self,
        n_mels: int = 100,
        num_subbands: int = 20,
        window: int = 32,
        num_harmonics: int = 12,
        channels: List[int] = [32, 64, 128, 256],
        use_harmonic: bool = True,
    ):
        super().__init__()
        self.spec = SpecDiscriminator(channels=channels)
        self.cross = CrossBandDiscriminator(
            n_mels=n_mels,
            num_subbands=num_subbands,
            window=window,
            channels=channels,
        )
        self.use_harmonic = use_harmonic
        self.harm = (
            HarmonicDiscriminator(num_harmonics=num_harmonics, channels=channels)
            if use_harmonic
            else None
        )

    @property
    def view_names(self) -> List[str]:
        return ["spec", "cross"] + (["harm"] if self.use_harmonic else [])

    def forward(
        self,
        mel: torch.Tensor,
        mask: torch.Tensor,
        cutoff_hz: torch.Tensor,
        centers: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        voiced: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, dict]:
        """
        Returns:
            ``(logits, features)``: dicts keyed by view name.
        """
        logits, features = {}, {}
        logits["spec"], features["spec"] = self.spec(mel)
        logits["cross"], features["cross"] = self.cross(mel, mask)
        if self.use_harmonic:
            assert f0 is not None and voiced is not None
            logits["harm"], features["harm"] = self.harm(
                mel, f0, voiced, cutoff_hz, centers
            )
        return logits, features
