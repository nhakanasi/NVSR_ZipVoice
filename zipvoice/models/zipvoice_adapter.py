#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Han Zhu)
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

"""Bandwidth-robust zero-shot TTS by a ControlNet-style side branch.

This is the second design for the same problem -- generate full-band speech from
a prompt that is band-limited -- and it exists because the first one was measured
failing in a way its structure could not be talked out of. That model routed a
frequency-conditioning vector into the frozen backbone's residual stream, scaled
by a learned gate. Two things went wrong.

The gate never learned to close. Evaluated on the 30k checkpoint with a
realistic prompt spectrum, ``alpha_enc`` sat at exactly its ceiling, 0.98, at
every cutoff from 24 kHz down to 2 kHz, and the per-bin read-out gate varied by
0.011 across that whole range. So the branch ran at full strength at full band,
where by construction it has nothing to contribute, and the full-band arm lost
0.07 SIM-o and 1.3 points of word error against the released checkpoint.

And it wrote everywhere. The branch's business is the bins the prompt does not
observe; the observed bins are already carried correctly by the frozen backbone,
which scores 2.79% word error from a 4 kHz prompt on its own. Every arm measured
so far let the correction touch all 100 mel bins, and every one of them raised
word error.

Both faults are removed here structurally rather than left to a gate:

* The backbone is never perturbed at its input. It runs on exactly the tensors
  it was pretrained on, whatever the prompt's bandwidth, and the branch may only
  add to the velocity it predicts.
* That addition is multiplied by the band mask, which is identically zero when
  the prompt is full band. Full band is then not a behaviour the model has to
  learn but an algebraic identity: this model is the released checkpoint there,
  bit for bit, at every point in training.

What remains is the branch itself, after ControlNet. The condition -- the
band-limited mel and the mask saying which of its bins survived -- goes through
the frequency encoder as feature extractor, then a two-layer convolutional
encoder, and is cross-attended by the backbone's own hidden output as query.
Each attention block's output passes a 1x1 convolution and is summed back into
the running state; a final linear reads that state out to one velocity value per
mel bin. There is no gate, no F0 path and no long-term average spectrum: the
encoder already pools whatever the mel contains, and the branch's strength is
now decided by the band mask and by the optimizer rather than by a scalar.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from zipvoice.models.modules.band_adapter import (
    BandAdapter,
    F0Encoder,
    FrequencyEncoder,
    RateEncoder,
)
from zipvoice.models.modules.bwe import (
    LOG_MEL_FLOOR,
    band_limit,
    mel_bin_centers,
    sample_cutoff,
)
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, make_pad_mask


def adapter_kwargs(model_config: Dict, **overrides) -> Dict:
    """The ``adapter`` section of a model config, with command-line overrides.

    The conditioning streams are a property of a training *arm* rather than of
    the architecture: the ablation runs several finetunes off one checkpoint
    against one config file. So every script that builds a
    :class:`ZipVoiceAdapter` takes the same flags, and any whose value is None
    leaves the config's own value alone. The flags have to be given to
    inference and to model averaging as well as to training, because a
    checkpoint carrying a stream's weights will not load strictly into a model
    built without it -- which is the desired failure, loudly rather than
    silently.
    """
    kwargs = dict(model_config.get("adapter", {}))
    for name, value in overrides.items():
        if value is not None:
            kwargs[name] = value
    return kwargs


class ZipVoiceAdapter(ZipVoice):
    """ZipVoice with a frozen backbone and a bandwidth-extension side branch."""

    def __init__(
        self,
        freq_dim: int = 64,
        freq_num_layers: int = 2,
        freq_num_heads: int = 4,
        freq_bin_out: int = 4,
        adapter_dim: int = 256,
        adapter_layers: int = 2,
        adapter_heads: int = 4,
        adapter_kernel: int = 5,
        write_mask_blur: int = 5,
        sampling_rate: int = 24000,
        feat_scale: float = 0.1,
        band_transition_bins: float = 3.0,
        band_floor_jitter: float = 0.1,
        band_noise_std: float = 0.0,
        cutoff_min: float = 1000.0,
        cutoff_max: float = 12000.0,
        curriculum_beta: float = 4.0,
        curriculum_rho: float = 0.7,
        full_band_prob: float = 0.2,
        use_f0_cond: bool = False,
        f0_cond_dim: int = 64,
        use_rate_cond: bool = False,
        rate_cond_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Degradation settings, shared verbatim with the bandwidth-extension
        # baseline, so both see the same simulated band-limiting and their
        # numbers are comparable.
        self.register_buffer(
            "mel_centers",
            mel_bin_centers(n_mels=self.feat_dim, sampling_rate=sampling_rate),
            persistent=False,
        )
        # ``features`` reach the model already multiplied by ``feat_scale``, so
        # the empty-bin floor has to be placed in the same units.
        self.mel_floor = LOG_MEL_FLOOR * feat_scale
        self.band_transition_bins = band_transition_bins
        self.band_floor_jitter = band_floor_jitter
        self.band_noise_std = band_noise_std
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max
        self.curriculum_beta = curriculum_beta
        self.curriculum_rho = curriculum_rho
        # A fraction of every batch is left full-band. The branch is inert there
        # by construction, so this costs it no capacity; what it buys is that the
        # *loss* sees the full-band case, which is the only way training can
        # notice if some later change breaks the inertness.
        self.full_band_prob = full_band_prob

        decoder_dim = self.fm_decoder.in_proj.out_features
        # The feature extractor reads the condition mel and the observed-bin
        # mask, and nothing else. It does not read the noisy state: the state
        # reaches the branch through the backbone's hidden output, which the
        # adapter cross-attends to, so reading it here would only duplicate it.
        # It does not read a long-term average spectrum either. That input was
        # added to the earlier model because its attention runs over bins inside
        # a single frame and so could not form any quantity pooled over time;
        # here the convolutional encoder that follows has a temporal receptive
        # field, and the cross-attention pools over the whole utterance, so the
        # recording chain's signature is derivable from the mel the branch is
        # already given.
        self.freq_encoder = FrequencyEncoder(
            num_bins=self.feat_dim,
            dim=freq_dim,
            num_layers=freq_num_layers,
            num_heads=freq_num_heads,
            bin_out=freq_bin_out,
            out_dim=adapter_dim,
        )
        # Optional extra conditioning streams. Their outputs are concatenated
        # onto the frequency encoder's along the channel axis, in this fixed
        # order, so each keeps channels of its own rather than competing for the
        # frequency encoder's; ``BandAdapter`` projects the result back to
        # ``adapter_dim`` at its entrance, and that projection is the identity
        # on the frequency encoder's channels, so a finetune from a checkpoint
        # trained without these streams starts at that checkpoint's function.
        self.f0_encoder = F0Encoder(out_dim=f0_cond_dim) if use_f0_cond else None
        self.rate_encoder = (
            RateEncoder(out_dim=rate_cond_dim) if use_rate_cond else None
        )
        cond_dim = adapter_dim
        if self.f0_encoder is not None:
            cond_dim += f0_cond_dim
        if self.rate_encoder is not None:
            cond_dim += rate_cond_dim
        self.adapter = BandAdapter(
            dim=adapter_dim,
            out_bins=self.feat_dim,
            query_dim=decoder_dim,
            cond_dim=cond_dim,
            num_layers=adapter_layers,
            num_heads=adapter_heads,
            kernel_size=adapter_kernel,
        )

        # How far the branch is allowed to write; see the module docstring. The
        # mask is blurred along the frequency axis first. ``band_limit`` returns
        # a transition only two bins wide at a 4 kHz cutoff -- 4087 to 4227 Hz --
        # and switching a correction on that abruptly risks a shelf at the
        # crossover. Blurring widens it without moving it, and cannot weaken the
        # full-band guarantee: the mean of a window of zeros is zero.
        self.write_mask_blur = write_mask_blur

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
        """Repeat a tensor along dim 0 to match a batch grown by guidance."""
        if x.size(0) == batch:
            return x
        assert batch % x.size(0) == 0, (batch, x.shape)
        return x.repeat(batch // x.size(0), *([1] * (x.dim() - 1)))

    def _write_mask(self, band_mask: torch.Tensor) -> torch.Tensor:
        """The bins the branch may write, from the bins the prompt lacks."""
        k = int(self.write_mask_blur)
        if k <= 1:
            return band_mask
        return F.avg_pool1d(
            band_mask, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False
        )

    def _band_limit(
        self, features: torch.Tensor, cutoff_hz: torch.Tensor, randomize: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Band-limit a mel with this model's degradation settings.

        ``randomize`` applies the training-time floor jitter and dead-band
        noise; inference leaves them off, since a real band-limited prompt
        already carries whatever its own encoder left behind.
        """
        return band_limit(
            features,
            cutoff_hz,
            centers=self.mel_centers,
            floor=self.mel_floor,
            transition_bins=self.band_transition_bins,
            floor_jitter=self.band_floor_jitter if randomize else 0.0,
            noise_std=self.band_noise_std if randomize else 0.0,
        )

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        band_mask: Optional[torch.Tensor] = None,
        cutoff_hz: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Velocity from the frozen backbone plus the branch's band residual.

        Args:
            band_mask: how missing each mel bin is in ``speech_condition``, of
                shape (B, 1, F), soft on the transition. None declares the
                prompt full band, in which case the branch contributes exactly
                zero.
            cutoff_hz: the prompt's cutoff in Hz, of shape (B,), for the rate
                stream. None means full band, which is ``cutoff_max``.
            prompt_mask: True at frames carrying prompt, of shape (B, T), for
                the F0 stream. None means every non-padded frame.

        The other arguments are as in the base class.
        """
        batch, num_frames, _ = xt.shape
        device = xt.device

        if band_mask is None:
            band_mask = torch.zeros(batch, 1, self.feat_dim, device=device)
        # Classifier-free guidance stacks the unconditional and conditional
        # halves into one batch inside the solver, but only touches the
        # arguments it knows about. The mask has to follow, and it is the same
        # for both halves: what guidance varies is the condition content, not
        # which bins the prompt happened to observe.
        band_mask = self._expand_to_batch(band_mask, batch)

        base = torch.cat([xt, text_condition, speech_condition], dim=2)

        t_arg = t
        assert t_arg.dim() in (0, 3)
        while t_arg.dim() > 1 and t_arg.size(-1) == 1:
            t_arg = t_arg.squeeze(-1)
        if t_arg.dim() == 0:
            t_arg = t_arg.repeat(batch)

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(batch)

        vt, hidden = self.fm_decoder(
            x=base,
            t=t_arg,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            return_hidden=True,
        )

        # Streams are concatenated in the fixed order the constructor sized
        # ``cond_dim`` for: frequency encoder, then F0, then rate.
        streams = [self.freq_encoder(speech_condition, band_mask)]
        if self.f0_encoder is not None:
            if prompt_mask is None:
                prompt_mask = (
                    torch.ones(batch, num_frames, dtype=torch.bool, device=device)
                    if padding_mask is None
                    else ~padding_mask
                )
            streams.append(
                self.f0_encoder(
                    speech_condition,
                    self._expand_to_batch(prompt_mask, batch),
                    self.mel_centers,
                )
            )
        if self.rate_encoder is not None:
            if cutoff_hz is None:
                cutoff_hz = torch.full((batch,), self.cutoff_max, device=device)
            streams.append(
                self.rate_encoder(self._expand_to_batch(cutoff_hz, batch), num_frames)
            )
        cond_feature = torch.cat(streams, dim=-1) if len(streams) > 1 else streams[0]
        residual = self.adapter(hidden, cond_feature, padding_mask)
        residual = residual * self._write_mask(band_mask)
        # Telemetry only; read by the training loop in place of the gate value
        # the earlier model logged, so that "how hard is the branch pushing"
        # stays visible on the training curves.
        self._residual_rms = residual.detach().pow(2).mean().sqrt()
        vt = vt + residual

        assert vt.shape[1] == num_frames, (vt.shape, num_frames)
        return vt

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
        cutoff_hz: Optional[torch.Tensor] = None,
        curriculum_step: int = 0,
        curriculum_total: int = 0,
        recon_loss_weight: float = 1.0,
        **unused,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for training.

        Two terms, and only two. The first is ZipVoice's own flow-matching loss
        over the generated span, unchanged, which keeps the model a
        text-to-speech model.

        The second is a spectrogram reconstruction loss. The velocity implies an
        estimate of the endpoint, ``x1_hat = x_t + (1 - t) * v_t``, and that
        estimate is a mel -- the quantity a vocoder is actually handed -- so the
        error is measured there rather than on the velocity. It is charged on the
        bins the band-limiting removed, over every valid frame, which is exactly
        the set of values the branch is allowed to write: the observed bins are
        the frozen backbone's business and the branch cannot touch them.

        L1 rather than squared error. Squared error is minimised by the
        conditional mean, and the conditional mean of a high band is flatter than
        any high band ever is; that over-smoothing is audible as a muffled upper
        register and is most of the MOS gap to a cascade at 4 kHz. L1 is
        minimised by the conditional median, which is less eager to average
        competing hypotheses, and it is what neural vocoders and bandwidth
        extenders are trained with for the same reason.

        Returns the two loss terms and an ``aux`` dict. ``aux`` carries the
        branch's residual rms, which is telemetry only, together with the
        endpoint estimate, the target, each utterance's generated span and ``t``.
        Those last four exist so the training script can charge additional losses
        on the generated audio without the modules that compute them becoming
        part of the model: anything attached here would enter the checkpoint and
        the averaged model, and the checkpoint has to stay the released tensors
        plus the branch.
        """
        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        if cutoff_hz is None:
            cutoff_hz = sample_cutoff(
                batch_size=features.size(0),
                step=curriculum_step,
                total_steps=curriculum_total,
                f_min=self.cutoff_min,
                f_max=self.cutoff_max,
                beta=self.curriculum_beta,
                rho=self.curriculum_rho,
                full_band_prob=self.full_band_prob,
                device=features.device,
            )
        elif not torch.is_tensor(cutoff_hz):
            cutoff_hz = torch.full(
                (features.size(0),), float(cutoff_hz), device=features.device
            )
        masked, band_mask = self._band_limit(features, cutoff_hz, randomize=True)

        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        # The condition carries the band-limited mel, not a restored one: the
        # model is meant to learn to live with the degradation, not to have it
        # repaired upstream.
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, masked)

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
            band_mask=band_mask,
            cutoff_hz=cutoff_hz,
            prompt_mask=~speech_condition_mask & ~padding_mask,
        )

        valid = ~padding_mask
        span_mask = speech_condition_mask & valid
        fm_loss = torch.mean((vt[span_mask] - ut[span_mask]) ** 2)

        # Spectrogram reconstruction on the endpoint estimate, over the bins the
        # branch may write. ``band_mask`` is the soft ramp ``band_limit``
        # returns, so a bin on the transition slope is weighted by how much of it
        # was actually removed rather than counted as wholly present or wholly
        # absent, and normalising by the total weight makes a narrow cutoff and a
        # wide one contribute equally instead of letting 4 kHz dominate the
        # gradient simply by having more bins to invent.
        x1_hat = xt + (1.0 - t) * vt
        weight = valid.unsqueeze(-1).to(vt.dtype) * band_mask  # (B, T, F)
        denom = weight.sum().clamp(min=1.0)
        recon_loss = ((x1_hat - features).abs() * weight).sum() / denom
        recon_loss = recon_loss * recon_loss_weight

        # ``condition_time_mask`` masks one contiguous window per utterance, so
        # the span is recovered from the mask itself rather than plumbed out of
        # it: the first masked frame, and that plus the count.
        span_len = speech_condition_mask.sum(dim=1)
        starts = speech_condition_mask.float().argmax(dim=1)
        aux = {
            "residual_rms": self._residual_rms,
            "x1_hat": x1_hat,
            "features": features,
            "span_starts": starts,
            "span_ends": starts + span_len,
            "t": t.reshape(-1),
        }
        return fm_loss, recon_loss, aux

    def set_trainable(self, mode: str) -> None:
        """Select which parameters the optimiser sees.

        ``adapter`` trains only the branch, leaving every released tensor absent
        from the optimiser, which is what makes the result mergeable back onto
        the released checkpoint. ``decoder`` additionally releases the
        flow-matching decoder, and ``all`` releases everything; both exist to
        measure what the restraint costs.
        """
        assert mode in ("adapter", "decoder", "all"), mode
        if mode == "all":
            self.requires_grad_(True)
            return
        self.requires_grad_(False)
        self.freq_encoder.requires_grad_(True)
        for stream in (self.f0_encoder, self.rate_encoder):
            if stream is not None:
                stream.requires_grad_(True)
        self.adapter.requires_grad_(True)
        if mode == "decoder":
            self.fm_decoder.requires_grad_(True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: torch.Tensor,
        prompt_features_lens: torch.Tensor,
        features_lens: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
        cutoff_hz: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate acoustic features.

        ``cutoff_hz`` is the prompt's Nyquist frequency, which the inference
        script derives from the sampling rate of the file it loaded. It tells the
        branch which bins of the prompt are genuinely observed as opposed to
        merely quiet, and which it may therefore write; passing None declares the
        prompt full band, which switches the branch off entirely.
        """
        assert duration in ["real", "predict"]

        if duration == "predict":
            (
                text_condition,
                padding_mask,
            ) = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
            )
        else:
            assert features_lens is not None
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )
        batch_size, num_frames, _ = text_condition.shape
        device = text_condition.device

        speech_condition = F.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B, T, F)

        # False means speech condition positions.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        cutoff = torch.full(
            (batch_size,),
            self.cutoff_max if cutoff_hz is None else float(cutoff_hz),
            device=device,
        )
        if cutoff_hz is None:
            band_mask = torch.zeros(batch_size, 1, self.feat_dim, device=device)
        else:
            # The prompt arrives already band-limited by its own recording rate,
            # so only the mask is wanted here; re-imposing the floor as well
            # would be a second degradation on top of the real one.
            _, band_mask = self._band_limit(speech_condition, cutoff, randomize=False)

        x0 = torch.randn(
            batch_size, num_frames, prompt_features.size(-1), device=device
        )

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            band_mask=band_mask,
            cutoff_hz=cutoff,
            # Same polarity as training's ``condition_time_mask``: True is the
            # span being generated, so the prompt is its complement.
            prompt_mask=~speech_condition_mask & ~padding_mask,
        )

        x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
        x1_prompt = torch.zeros(
            batch_size,
            prompt_features.size(1),
            x1.size(-1),
            device=device,
            dtype=x1.dtype,
        )
        x1_wo_prompt = torch.zeros(
            batch_size,
            (num_frames - prompt_features_lens.min()).item(),
            x1.size(-1),
            device=device,
            dtype=x1.dtype,
        )
        for i in range(batch_size):
            x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                i,
                prompt_features_lens[i] : prompt_features_lens[i]
                + x1_wo_prompt_lens[i],
                :,
            ]
            x1_prompt[i, : prompt_features_lens[i], :] = x1[
                i, : prompt_features_lens[i], :
            ]

        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens
