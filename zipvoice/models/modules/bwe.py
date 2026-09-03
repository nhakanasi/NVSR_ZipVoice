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
Mel-domain bandwidth extension (BWE) for ZipVoice speech prompts.

The module lets ZipVoice accept a speech prompt recorded at an arbitrary sampling
rate: the prompt is resampled to 24 kHz as usual, which leaves the mel bins above
the original Nyquist frequency empty, and this module fills them back in.

Contents:
  * mel filterbank center frequencies matching ``zipvoice.utils.feature.VocosFbank``
  * the "Easy-to-Balanced" cutoff curriculum of AnyBand (arXiv 2608.00572, Eq. 5)
  * ``band_limit``: simulates a band-limited prompt by masking high mel bins
  * ``lsd_loss`` / ``ild_loss`` / ``ndl_loss``: the spectral reconstruction
    objective of HRTFformer (arXiv 2510.01891), adapted to the mel grid
  * ``NVSRResUNet``: an NVSR-style ResUNet that predicts the missing band
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# log(1e-7): the value a fully empty mel bin takes, given that VocosFbank clamps
# the linear mel at 1e-7 before taking the natural log.
# See zipvoice/utils/feature.py.
LOG_MEL_FLOOR = math.log(1e-7)


def _hz_to_mel(freq: float) -> float:
    """HTK mel scale, matching torchaudio's default ``mel_scale="htk"``."""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_bin_centers(
    n_mels: int = 100,
    sampling_rate: int = 24000,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Center frequencies (Hz) of each mel filter, for the filterbank built by
    ``torchaudio.transforms.MelSpectrogram`` with its default HTK mel scale.

    torchaudio spaces ``n_mels + 2`` points uniformly on the mel axis and uses
    points ``1 .. n_mels`` as the triangle centers.

    Returns:
        A tensor of shape ``(n_mels,)``, monotonically increasing.
    """
    if f_max is None:
        f_max = sampling_rate / 2
    m_min, m_max = _hz_to_mel(f_min), _hz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    return _mel_to_hz(m_pts[1:-1])


def hz_to_mel_bin(freq: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Map frequencies in Hz to fractional mel-bin indices by linear interpolation
    between filter centers.

    Args:
        freq: frequencies in Hz, any shape.
        centers: output of :func:`mel_bin_centers`, shape ``(n_mels,)``.

    Returns:
        Fractional bin indices with the same shape as ``freq``, clamped to
        ``[0, n_mels - 1]``.
    """
    n_mels = centers.numel()
    centers = centers.to(device=freq.device, dtype=freq.dtype)
    # Number of centers strictly below each frequency, i.e. the index of the
    # interval the frequency falls into.
    idx = torch.searchsorted(centers, freq.contiguous().reshape(-1)).reshape(freq.shape)
    lo = (idx - 1).clamp(0, n_mels - 1)
    hi = idx.clamp(0, n_mels - 1)
    c_lo, c_hi = centers[lo], centers[hi]
    frac = torch.where(c_hi > c_lo, (freq - c_lo) / (c_hi - c_lo + 1e-9),
                       torch.zeros_like(freq))
    return (lo + frac.clamp(0.0, 1.0)).clamp(0.0, n_mels - 1)


def curriculum_lambda(step: int, total_steps: int, rho: float = 0.7) -> float:
    """
    Cosine annealing weight of the uniform component in the Easy-to-Balanced
    cutoff curriculum (AnyBand Eq. 5)::

        lambda_s = [1 - cos(pi * min(s / (rho * S), 1))] / 2

    Starts at 0 (fully exponential, biased towards high cutoffs, i.e. easy
    less-underdetermined cases) and reaches 1 (uniform over the whole cutoff
    range) at ``rho * total_steps``.
    """
    if total_steps <= 0 or rho <= 0:
        return 1.0
    progress = min(step / (rho * total_steps), 1.0)
    return (1.0 - math.cos(math.pi * progress)) / 2.0


def sample_cutoff(
    batch_size: int,
    step: int,
    total_steps: int,
    f_min: float = 1000.0,
    f_max: float = 12000.0,
    beta: float = 4.0,
    rho: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sample one cutoff frequency per utterance from the Easy-to-Balanced
    curriculum distribution of AnyBand (Eq. 5)::

        p_s(f~) = (1 - lambda_s) * beta * e^(beta * f~) / (e^beta - 1) + lambda_s

    over the normalized cutoff ``f~ = (f_c - f_min) / (f_max - f_min)``.

    Sampled by picking the mixture component and then inverting its CDF: the
    exponential branch has ``CDF(f~) = (e^(beta f~) - 1) / (e^beta - 1)``, so
    ``f~ = log(1 + u * (e^beta - 1)) / beta``.

    Set ``rho <= 0`` (or ``total_steps <= 0``) for plain uniform sampling, which
    is what the adversarial refinement stage uses.

    Returns:
        Cutoff frequencies in Hz, shape ``(batch_size,)``.
    """
    lam = curriculum_lambda(step, total_steps, rho)
    u = torch.rand(batch_size, device=device)
    exp_branch = torch.log1p(u * math.expm1(beta)) / beta
    use_uniform = torch.rand(batch_size, device=device) < lam
    f_tilde = torch.where(use_uniform, u, exp_branch)
    return f_min + f_tilde * (f_max - f_min)


def band_limit(
    features: torch.Tensor,
    cutoff_hz: torch.Tensor,
    centers: torch.Tensor,
    floor: float = LOG_MEL_FLOOR,
    transition_bins: float = 3.0,
    floor_jitter: float = 0.0,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate a band-limited prompt by pushing the mel bins above ``cutoff_hz``
    down to the empty-bin floor.

    A real 16 kHz -> 24 kHz resampling does not produce a brick wall: there is a
    transition band, and the dead region sits at a jittery noise floor rather
    than exactly at the clamp value. ``transition_bins``, ``floor_jitter`` and
    ``noise_std`` randomize those away so the model does not overfit to an
    idealized cutoff.

    Args:
        features: log-mel of shape ``(B, T, F)``, already multiplied by
            ``params.feat_scale``.
        cutoff_hz: cutoff frequency per utterance, shape ``(B,)``.
        centers: mel filter center frequencies, shape ``(F,)``.
        floor: value of an empty log-mel bin, in the same scale as ``features``
            (i.e. already multiplied by ``feat_scale``).
        transition_bins: width in mel bins of the raised-cosine roll-off. The
            actual width is drawn uniformly from ``[0.5, 1.5]`` times this.
        floor_jitter: half-width of the per-utterance uniform jitter on ``floor``.
        noise_std: standard deviation of Gaussian noise added in the dead band.

    Returns:
        A tuple ``(masked, mask)`` where ``masked`` has the shape of ``features``
        and ``mask`` has shape ``(B, 1, F)`` with 1.0 for fully missing bins,
        0.0 for fully observed bins and a ramp in between.
    """
    b, _, n_mels = features.shape
    device, dtype = features.device, features.dtype

    cutoff_bin = hz_to_mel_bin(cutoff_hz.to(dtype), centers).view(b, 1)  # (B, 1)
    width = transition_bins * (
        0.5 + torch.rand(b, 1, device=device, dtype=dtype)
    )  # (B, 1) in [0.5, 1.5] * transition_bins
    width = width.clamp(min=1e-3)

    bins = torch.arange(n_mels, device=device, dtype=dtype).view(1, n_mels)
    # 0 at and below the cutoff bin, 1 once ``width`` bins above it, raised
    # cosine in between. The ramp starts *at* the cutoff rather than straddling
    # it so that a cutoff at or above the top filter center leaves the mel
    # untouched, which is what a full-band prompt has to reduce to.
    ramp = ((bins - cutoff_bin) / width).clamp(0.0, 1.0)
    mask = (1.0 - torch.cos(math.pi * ramp)) / 2.0  # (B, F)
    mask = mask.unsqueeze(1)  # (B, 1, F)

    fill = torch.full((b, 1, 1), float(floor), device=device, dtype=dtype)
    if floor_jitter > 0:
        fill = fill + (torch.rand_like(fill) * 2.0 - 1.0) * floor_jitter
    if noise_std > 0:
        fill = fill + torch.randn_like(features) * noise_std

    masked = features * (1.0 - mask) + fill * mask
    return masked, mask


def _to_db(features: torch.Tensor, feat_scale: float) -> torch.Tensor:
    """
    Convert a scaled natural-log mel back to decibels.

    ``features`` is ``feat_scale * log(magnitude)``, so the magnitude in dB is
    ``20 * log10(magnitude) = features * 20 / (ln(10) * feat_scale)``. Working in
    dB is what makes the losses below comparable to the numbers reported in the
    bandwidth-extension literature.
    """
    return features * (20.0 / (math.log(10.0) * feat_scale))


def lsd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    feat_scale: float,
) -> torch.Tensor:
    """
    Log-spectral distance, HRTFformer Eq. 9 (arXiv 2510.01891) on the mel grid.

    Root-mean-square of the dB error across mel bins within a frame, then a plain
    mean across frames. The inner square root is the whole point: it makes every
    frame contribute equally regardless of how badly it is reconstructed, so a few
    hard frames cannot dominate the gradient the way they do under a plain MSE.

    Args:
        pred: predicted log-mel, ``(B, T, F)``, in ``feat_scale`` units.
        target: reference log-mel, same shape and units.
        weight: per-bin weight in ``[0, 1]``, ``(B, T, F)``; frames whose weights
            are all zero are dropped from the outer mean.
        feat_scale: the ``--feat-scale`` the features were multiplied by.

    Returns:
        A scalar.
    """
    err = (_to_db(pred, feat_scale) - _to_db(target, feat_scale)) ** 2
    denom = weight.sum(dim=-1)  # (B, T)
    # The epsilon under the root keeps the gradient of sqrt finite at zero error.
    per_frame = torch.sqrt((err * weight).sum(dim=-1) / denom.clamp(min=1e-8) + 1e-12)
    valid = (denom > 0).to(per_frame.dtype)
    return (per_frame * valid).sum() / valid.sum().clamp(min=1.0)


def ild_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    band_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    feat_scale: float,
    subband_bins: int = 5,
) -> torch.Tensor:
    """
    Inter-band level difference, adapted from HRTFformer Eq. 10.

    The paper's ILD is the left-minus-right level ratio of a binaural pair. A mono
    mel has no second channel, so the analogous pair of related levels here is the
    restored band against the observed band: the loss asks for the *balance*
    between what was measured and what was invented to be right, which is the
    error a bandwidth extender actually makes -- output that is too dull or too
    bright overall while every individual bin looks plausible.

    Levels are taken per subband of ``subband_bins`` mel bins, matching the
    subband layout of ``CrossBandDiscriminator``. Note that
    ``NVSRResUNet.forward`` copies fully observed bins through bit-exact, so the
    observed reference level cancels between prediction and target and the term
    reduces to a subband energy-envelope loss. That is deliberate: it constrains
    the coarse shape of the restored band, where :func:`lsd_loss` constrains the
    per-bin detail.

    Args:
        pred: predicted log-mel, ``(B, T, F)``, in ``feat_scale`` units.
        target: reference log-mel, same shape and units.
        band_mask: ``(B, 1, F)``, 1 for restored bins, 0 for observed ones.
        frame_mask: ``(B, T, 1)``, 1 for real frames.
        feat_scale: the ``--feat-scale`` the features were multiplied by.
        subband_bins: mel bins per subband. Must divide the number of mel bins.

    Returns:
        A scalar.
    """
    b, t, n_mels = pred.shape
    assert n_mels % subband_bins == 0, (n_mels, subband_bins)
    n_sub = n_mels // subband_bins

    pred_db = _to_db(pred, feat_scale)
    target_db = _to_db(target, feat_scale)

    # Observed reference level: mean dB over the fully observed bins of a frame.
    obs = (1.0 - band_mask) * frame_mask  # (B, T, F)
    obs_denom = obs.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, T, 1)
    pred_ref = (pred_db * obs).sum(dim=-1, keepdim=True) / obs_denom
    target_ref = (target_db * obs).sum(dim=-1, keepdim=True) / obs_denom

    # Subband levels, weighted by how much of each bin was restored.
    w = (band_mask * frame_mask).reshape(b, t, n_sub, subband_bins)
    w_denom = w.sum(dim=-1)  # (B, T, n_sub)
    pred_sub = (pred_db.reshape(b, t, n_sub, subband_bins) * w).sum(dim=-1)
    target_sub = (target_db.reshape(b, t, n_sub, subband_bins) * w).sum(dim=-1)
    pred_sub = pred_sub / w_denom.clamp(min=1e-8)
    target_sub = target_sub / w_denom.clamp(min=1e-8)

    pred_ild = pred_sub - pred_ref
    target_ild = target_sub - target_ref

    valid = (w_denom > 0).to(pred.dtype)
    return ((pred_ild - target_ild).abs() * valid).sum() / valid.sum().clamp(min=1.0)


def ndl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    feat_scale: float,
) -> torch.Tensor:
    """
    Neighbor dissimilarity loss, HRTFformer Eq. 11, on the time-frequency grid.

    For every point, take its deviation from the mean of its four neighbours
    (earlier frame, later frame, bin below, bin above) and match that deviation
    between prediction and target. Absolute level is free; only local structure is
    penalised, which is what keeps a reconstruction from going flat and smeared
    across the restored band. Edges use replication padding.

    Args:
        pred: predicted log-mel, ``(B, T, F)``, in ``feat_scale`` units.
        target: reference log-mel, same shape and units.
        weight: per-bin weight in ``[0, 1]``, ``(B, T, F)``.
        feat_scale: the ``--feat-scale`` the features were multiplied by.

    Returns:
        A scalar.
    """
    # The deviation operator is linear, so applying it to the error is the same
    # as differencing the two deviation maps, and costs one pad instead of two.
    err = (_to_db(pred, feat_scale) - _to_db(target, feat_scale)).unsqueeze(1)
    padded = torch.nn.functional.pad(err, (1, 1, 1, 1), mode="replicate")
    neighbors = (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
    ) / 4.0
    deviation = (err - neighbors).squeeze(1)  # (B, T, F)
    return (deviation.square() * weight).sum() / weight.sum().clamp(min=1.0)


class ResConvBlock(nn.Module):
    """Two 3x3 convolutions with a 1x1 projection shortcut, as in NVSR's ResUNet."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        groups = math.gcd(num_groups, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class NVSRResUNet(nn.Module):
    """
    NVSR-style ResUNet operating on the log-mel "image".

    The network sees the band-limited mel stacked with the band mask, predicts a
    residual, and returns the hard composite of AnyBand Eq. 10::

        M_fb = (1 - mask) * M_masked + mask * M_hat

    so the observed low band is copied through bit-exact -- this is also NVSR's
    "replacement" post-processing, done inside the module rather than after it.
    """

    def __init__(
        self,
        n_mels: int = 100,
        channels: List[int] = [32, 64, 128, 256],
        use_mask_channel: bool = True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.use_mask_channel = use_mask_channel
        self.num_stages = len(channels)
        self.stride = 2**self.num_stages

        in_channels = 2 if use_mask_channel else 1
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoders.append(ResConvBlock(prev, ch))
            self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev = ch

        self.bottleneck = ResConvBlock(prev, prev)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1))
            # Skip connection doubles the channel count.
            self.decoders.append(ResConvBlock(ch * 2, ch))
            prev = ch

        self.head = nn.Conv2d(prev, 1, 1)
        # Start as a no-op so that early training does not corrupt the prompt.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked: band-limited log-mel, shape ``(B, T, F)``.
            mask: band mask from :func:`band_limit`, shape ``(B, 1, F)``, 1.0
                where the band is missing.

        Returns:
            The restored full-band log-mel, shape ``(B, T, F)``.
        """
        b, t, f = masked.shape
        assert f == self.n_mels, (f, self.n_mels)

        x = masked.unsqueeze(1)  # (B, 1, T, F)
        if self.use_mask_channel:
            x = torch.cat([x, mask.unsqueeze(1).expand(b, 1, t, f)], dim=1)

        pad_t = (-t) % self.stride
        pad_f = (-f) % self.stride
        if pad_t or pad_f:
            x = F.pad(x, (0, pad_f, 0, pad_t), mode="replicate")

        skips = []
        h = self.stem(x)
        for encoder, down in zip(self.encoders, self.downs):
            h = encoder(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            h = decoder(torch.cat([h, skip], dim=1))

        residual = self.head(h)[:, 0, :t, :f]  # (B, T, F)
        predicted = masked + residual
        return masked * (1.0 - mask) + predicted * mask
