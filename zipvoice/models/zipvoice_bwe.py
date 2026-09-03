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

from typing import List, Optional, Tuple

import torch

from zipvoice.models.modules.bwe import (
    LOG_MEL_FLOOR,
    NVSRResUNet,
    band_limit,
    mel_bin_centers,
)
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask


class ZipVoiceBWE(ZipVoice):
    """
    ZipVoice with a mel bandwidth extender in front of the speech condition.

    The extender restores the mel bins above the promptf's original Nyquist
    frequency, so a prompt recorded at any sampling rate can condition the model.
    It is trained jointly: the flow-matching target stays the clean full-band
    mel, so the flow-matching loss is itself a task-aware objective for the
    extender, on top of the reconstruction and adversarial losses applied to its
    output in the training script.
    """

    def __init__(
        self,
        bwe_channels: List[int] = [32, 64, 128, 256],
        bwe_use_mask_channel: bool = True,
        bwe_transition_bins: float = 3.0,
        bwe_floor_jitter: float = 0.1,
        bwe_noise_std: float = 0.02,
        feat_scale: float = 0.1,
        sampling_rate: int = 24000,
        **kwargs,
    ):
        """
        Args:
            bwe_channels: per-stage channel counts of the extender's ResUNet.
            bwe_use_mask_channel: feed the band mask as a second input channel.
            bwe_transition_bins: nominal width in mel bins of the simulated
                band-limiting roll-off.
            bwe_floor_jitter: half-width of the uniform jitter on the empty-bin
                floor during training.
            bwe_noise_std: standard deviation of the noise added in the dead band
                during training.
            feat_scale: the scale features are multiplied by before reaching the
                model (``params.feat_scale``); needed to place the empty-bin
                floor in the same units.
            sampling_rate: sampling rate the mel was computed at.
        """
        super().__init__(feat_dim=kwargs.pop("feat_dim", 100), **kwargs)

        self.bwe = NVSRResUNet(
            n_mels=self.feat_dim,
            channels=bwe_channels,
            use_mask_channel=bwe_use_mask_channel,
        )
        # Which part of the extender to skip at inference. "none" runs it
        # normally; "resunet" band-limits and re-floors the dead band but does
        # not predict into it; "full" is read by the inference path, which then
        # never calls restore() at all. Set from --bwe-bypass; the ablation is
        # what separates the extender's contribution from the fine-tuning's.
        self.bwe_bypass = "none"
        self.bwe_transition_bins = bwe_transition_bins
        self.bwe_floor_jitter = bwe_floor_jitter
        self.bwe_noise_std = bwe_noise_std
        self.mel_floor = LOG_MEL_FLOOR * feat_scale
        self.register_buffer(
            "mel_centers",
            mel_bin_centers(n_mels=self.feat_dim, sampling_rate=sampling_rate),
            persistent=False,
        )

    def restore(
        self,
        features: torch.Tensor,
        cutoff_hz: torch.Tensor,
        randomize: bool = False,
        use_extender: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Band-limit ``features`` at ``cutoff_hz`` and restore the missing band.

        Applied at inference too, on a prompt that is already band-limited by
        its own recording sampling rate: re-imposing the same floor keeps the
        dead band identical to what the extender saw in training, instead of
        whatever residue the resampler happened to leave there.

        Args:
            features: log-mel, shape ``(B, T, F)``, already scaled by
                ``feat_scale``.
            cutoff_hz: cutoff per utterance, shape ``(B,)``.
            randomize: apply the training-time floor jitter and dead-band noise.
                Leave False at inference.
            use_extender: run the ResUNet. False returns the band-limited mel
                itself, which is the ablation that holds the dead-band floor
                fixed and removes only the prediction.

        Returns:
            ``(restored, mask)``: the full-band mel of shape ``(B, T, F)``, and
            the band mask of shape ``(B, 1, F)``.
        """
        masked, mask = band_limit(
            features=features,
            cutoff_hz=cutoff_hz,
            centers=self.mel_centers,
            floor=self.mel_floor,
            transition_bins=self.bwe_transition_bins,
            floor_jitter=self.bwe_floor_jitter if randomize else 0.0,
            noise_std=self.bwe_noise_std if randomize else 0.0,
        )
        if not use_extender:
            return masked, mask
        return self.bwe(masked, mask), mask

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        cutoff_hz: Optional[torch.Tensor] = None,
        condition_drop_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model for training.

        Identical to :meth:`ZipVoice.forward` except that the speech condition is
        built from the bandwidth-extended mel rather than from ``features``
        directly. The flow-matching target is still the clean ``features``.

        Args:
            tokens: a list of list of token ids.
            features: the clean full-band acoustic features, with the shape
                (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape
                (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            cutoff_hz: the simulated band-limiting cutoff per utterance, shape
                (batch,). If None, the extender is bypassed and this behaves
                exactly like :meth:`ZipVoice.forward`.
            condition_drop_ratio: the ratio of dropped text condition.
        Returns:
            fm_loss: the flow-matching loss.
            mel_fb: the bandwidth-extended mel, shape (batch, seq_len, feat_dim).
            band_mask: the band mask, shape (batch, 1, feat_dim).
        """

        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        if cutoff_hz is None:
            mel_fb = features
            band_mask = torch.zeros(
                features.size(0), 1, features.size(2), device=features.device,
                dtype=features.dtype,
            )
        else:
            mel_fb, band_mask = self.restore(features, cutoff_hz, randomize=True)

        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, mel_fb)

        if condition_drop_ratio > 0.0:
            drop_mask = (
                torch.rand(text_condition.size(0), 1, 1).to(text_condition.device)
                > condition_drop_ratio
            )
            text_condition = text_condition * drop_mask

        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        return fm_loss, mel_fb, band_mask
