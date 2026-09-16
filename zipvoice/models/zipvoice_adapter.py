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

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipvoice.models.modules.band_adapter import (
    BandAdapter,
    FrequencyEncoder,
    RateEncoder,
    SpeakerEncoder,
)
from zipvoice.models.modules.bwe import (
    LOG_MEL_FLOOR,
    band_limit,
    mel_bin_centers,
    sample_cutoff,
)
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, make_pad_mask


# Tile sizes for the multi-resolution band loss, as ``(frames, mel bins)``.
# The feature rate is 93.75 fps, so these are roughly 85 ms, 340 ms and 1.4 s,
# over 5, 10 and 20 of the 100 mel bins. The fine end is deliberately not 1x1:
# that is ``recon_loss``, which this term exists to complement rather than
# duplicate.
_BAND_LOSS_TILES = ((8, 5), (32, 10), (128, 20))


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
        use_speaker_cond: bool = False,
        speaker_cond_dim: int = 128,
        use_rate_cond: bool = False,
        rate_cond_dim: int = 64,
        use_pretrained_spk: bool = False,
        pretrained_spk_dim: int = 256,
        pretrained_spk_cond_dim: int = 128,
        mask_xt: bool = False,
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
        # When enabled, the adapter hidden state is computed from a state whose
        # missing band is hidden. The base velocity still uses the normal flow
        # state, so the clean target remains supervision rather than adapter
        # input.
        self.mask_xt = mask_xt
        self.curriculum_beta = curriculum_beta
        self.curriculum_rho = curriculum_rho
        # Not a trained quantity and not part of the config: an inference dial,
        # set by the caller between synthesis runs to measure the identity /
        # naturalness trade the branch makes. See ``forward_fm_decoder``.
        self.residual_scale = 1.0
        # Inference switch for a branch trained with --band-fill. The branch
        # then emits a mel rather than a velocity, so its output must not join
        # the solver's; ``sample`` adds the last read-out to the solved mel.
        self.band_fill = False
        self._bf_state = None
        # The two halves of the scheduled dose. ``None`` keeps ``residual_scale``
        # flat in that variable, which is the behaviour every arm before this
        # was measured under. See ``forward_fm_decoder``.
        self.residual_scale_end = None
        self.residual_scale_hi = None
        # 0.0 means "an octave above this utterance's cutoff" rather than a
        # fixed frequency, because the cutoff is what the evaluation sweeps.
        self.residual_scale_split_hz = 0.0
        # Where ``residual_scale_top`` applies. 0.0 means the top mel bin.
        self.residual_scale_top_hz = 0.0
        # ``None`` makes the raised dose a flat step, which is what the split
        # meant before the per-octave sensitivity was measured.
        self.residual_scale_top = None
        # Trajectory decoupling, also inference-only. See
        # ``forward_fm_decoder``; the two buffers are per-``sample`` state and
        # are cleared there.
        self.decouple_residual = False
        # State-level projection, inference-only. See ``_projected_sample``.
        self.project_low_band = False
        # Filled by ``_projected_sample``: one row per ODE step, read by the
        # caller to see when the low-band divergence appears.
        self.project_trace = None
        # Set by ``forward`` around its decoder call when the drift term is on.
        # The term is a finite difference between two backbone forwards, and the
        # backbone has dropout and stochastic depth, so without pinning the
        # generator the second forward draws different masks and the difference
        # measures those instead of the perturbation. Measured on a random-init
        # model: two forwards of the *same* input differ by 1.2e-4 where a real
        # perturbation moves them 1.27e-4, so the signal-to-noise ratio is 1.07
        # and the term is noise.
        self._capture_rng = False
        self._rng_state = None
        self._decouple_delta = None
        self._decouple_prev = None
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
        self.speaker_encoder = (
            SpeakerEncoder(num_bins=self.feat_dim, out_dim=speaker_cond_dim)
            if use_speaker_cond
            else None
        )
        self.rate_encoder = (
            RateEncoder(out_dim=rate_cond_dim) if use_rate_cond else None
        )
        # The pretrained speaker stream. Unlike the three above, the encoder that
        # produces it is not part of this model: it is a frozen vocoder and a
        # frozen WeSpeaker ResNet34, which the training script and the inference
        # script each own, and only the embedding is passed in. Keeping them out
        # means a saved checkpoint is still the released tensors plus the branch,
        # and still merges back onto the released model.
        #
        # The embedding arrives with a validity flag appended, hence the + 1.
        self.pretrained_spk_proj = (
            nn.Linear(pretrained_spk_dim + 1, pretrained_spk_cond_dim)
            if use_pretrained_spk
            else None
        )
        if self.pretrained_spk_proj is not None:
            # Small, not zero, for the reason argued in ``band_adapter.py``:
            # ScaledAdam parametrises each tensor by its own norm and cannot
            # move one that starts at exactly zero.
            nn.init.normal_(self.pretrained_spk_proj.weight, std=1e-3)
            nn.init.zeros_(self.pretrained_spk_proj.bias)
        self.pretrained_spk_dim = pretrained_spk_dim
        cond_dim = adapter_dim
        if self.speaker_encoder is not None:
            cond_dim += speaker_cond_dim
        if self.rate_encoder is not None:
            cond_dim += rate_cond_dim
        if self.pretrained_spk_proj is not None:
            cond_dim += pretrained_spk_cond_dim
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
        speaker_embedding: Optional[torch.Tensor] = None,
        adapter_state: Optional[torch.Tensor] = None,
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
            speaker_embedding: the prompt's embedding under a frozen pretrained
                speaker encoder with its validity flag appended, of shape
                (B, D + 1); see ``SpeakerPerceptual.embed_prompt``, which owns
                the encoder and the argument that this carries nothing from
                above the cutoff. None with the stream enabled is a zero
                embedding and a zero flag, which is the same thing the encoder
                returns for a prompt too short to embed.

        The other arguments are as in the base class.
        """
        batch, num_frames, _ = xt.shape
        device = xt.device

        # Trajectory decoupling. The branch writes only above the cutoff, but
        # the solver carries those bins into the next step's ``x_t`` and the
        # frozen backbone reads the whole spectrum, so the branch's writes feed
        # back into the backbone's own low-band synthesis over the 16 steps.
        # That feedback is not something training ever posed: training is one
        # step with ``x_t`` built from ground-truth features, so the branch is
        # fitted as though the trajectory it rides were the true one.
        #
        # Measured, the feedback is where the damage is: lowpass every system
        # back at its cutoff and the adapter is still 0.115 below stock at
        # 4 kHz against a 0.122 full-band gap, so ~94% of the identity loss sits
        # in bins the branch never writes.
        #
        # So the branch's contribution is kept out of what the backbone sees.
        # ``_decouple_delta`` accumulates it; the backbone is evaluated at
        # ``x_t - delta``, which is the trajectory it would have followed alone,
        # while the solver keeps integrating the sum and therefore still returns
        # ``x_stock + delta`` at the end. Euler's step size is not visible here,
        # so it is recovered from consecutive ``t`` values on the next call.
        if self.decouple_residual:
            t_now = float(t.reshape(-1)[0])
            if self._decouple_delta is None:
                self._decouple_delta = torch.zeros_like(xt)
            elif self._decouple_prev is not None:
                t_prev, residual_prev = self._decouple_prev
                self._decouple_delta = self._decouple_delta + (
                    t_now - t_prev
                ) * residual_prev
            xt = xt - self._decouple_delta

        if band_mask is None:
            band_mask = torch.zeros(batch, 1, self.feat_dim, device=device)
        # Classifier-free guidance stacks the unconditional and conditional
        # halves into one batch inside the solver, but only touches the
        # arguments it knows about. The mask has to follow, and it is the same
        # for both halves: what guidance varies is the condition content, not
        # which bins the prompt happened to observe.
        band_mask = self._expand_to_batch(band_mask, batch)

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

        base = torch.cat([xt, text_condition, speech_condition], dim=2)

        # See ``_capture_rng``. Taken immediately before the decoder runs so the
        # drift term's second forward can replay this one's dropout draw.
        if self._capture_rng:
            self._rng_state = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(xt.device) if xt.is_cuda else None,
            )

        # Keep the stock path unchanged: the frozen backbone receives the
        # ordinary flow state and the degraded prompt, and its velocity remains
        # the stock model's prediction. The adapter uses a separate hidden-state
        # path so the clean target cannot reach the branch through ``x_t``.
        vt, hidden = self.fm_decoder(
            x=base,
            t=t_arg,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            return_hidden=True,
        )
        if self.mask_xt and band_mask.any():
            # ``band_mask`` marks bins absent from the prompt. Replace those
            # bins before the decoder produces the hidden state used by the
            # adapter; masking hidden channels afterward cannot identify
            # frequency content because the decoder has already mixed it.
            hidden_xt = xt * (1.0 - band_mask) + self.mel_floor * band_mask
            hidden_base = torch.cat(
                [hidden_xt, text_condition, speech_condition], dim=2
            )
            _, hidden = self.fm_decoder(
                x=hidden_base,
                t=t_arg,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                return_hidden=True,
            )

        # Streams are concatenated in the fixed order the constructor sized
        # ``cond_dim`` for: frequency encoder, speaker, rate, pretrained
        # speaker.
        streams = [self.freq_encoder(speech_condition, band_mask)]
        if self.speaker_encoder is not None:
            if prompt_mask is None:
                prompt_mask = (
                    torch.ones(batch, num_frames, dtype=torch.bool, device=device)
                    if padding_mask is None
                    else ~padding_mask
                )
            streams.append(
                self.speaker_encoder(
                    speech_condition, self._expand_to_batch(prompt_mask, batch)
                )
            )
        if self.rate_encoder is not None:
            if cutoff_hz is None:
                cutoff_hz = torch.full((batch,), self.cutoff_max, device=device)
            streams.append(
                self.rate_encoder(self._expand_to_batch(cutoff_hz, batch), num_frames)
            )
        if self.pretrained_spk_proj is not None:
            if speaker_embedding is None:
                speaker_embedding = torch.zeros(
                    batch, self.pretrained_spk_dim + 1, device=device, dtype=xt.dtype
                )
            # One vector for the whole utterance, broadcast over time. It says
            # whose voice this is, which does not vary along the utterance; the
            # frequency encoder's stream carries everything that does.
            spk = self.pretrained_spk_proj(
                self._expand_to_batch(speaker_embedding, batch).to(xt.dtype)
            )
            streams.append(spk.unsqueeze(1).expand(-1, num_frames, -1))
        cond_feature = torch.cat(streams, dim=-1) if len(streams) > 1 else streams[0]
        if self.band_fill and adapter_state is None:
            # Inference goes through the solver, which has no place to thread a
            # state through, so the branch carries its own across the steps of
            # one ``sample`` call. Reset there, not here.
            adapter_state = self._bf_state
        residual, self._last_adapter_state = self.adapter(
            hidden,
            cond_feature,
            padding_mask,
            state=adapter_state,
            return_state=True,
        )
        if self.band_fill:
            self._bf_state = self._last_adapter_state
        residual = residual * self._write_mask(band_mask)
        # Inference-time dose control, 1.0 during training. The branch trades
        # speaker similarity for naturalness -- it fills the missing band with
        # something plausible that is not quite this speaker -- and scaling the
        # residual walks that trade without retraining: 0.0 is the frozen model,
        # 1.0 is the branch as trained.
        #
        # A flat dose is a blunt instrument, and the sweep showed why: the
        # branch writes 0.121 rms on the training path and 0.068 on a state it
        # produced itself twelve steps in, so matching the two calls for a gain
        # near 1.8, and a flat 1.75 costs SIM-o 0.246 -> 0.156. The deficit and
        # the damage are not in the same place. The deficit is spread over
        # 3-12 kHz; the identity lives in the octave just above the cutoff,
        # where the formant structure the prompt half-shows is continued. And
        # it accumulates along the solve: early steps write into a state that
        # is still mostly noise, late steps into one whose band is already
        # short. So the dial is split in two, after the exposure-bias
        # literature, which corrects a train/sample norm mismatch with a
        # schedule rather than a constant:
        #
        #   ``residual_scale`` -> ``residual_scale_end``  linear in t
        #   ``residual_scale_hi`` above ``residual_scale_split_hz``
        #
        # Both default to the flat behaviour, so an unset model is the dial as
        # it was. The two factors multiply: the schedule says how hard to push
        # at this point on the trajectory, the split says where in frequency
        # the push is safe.
        gain = self.residual_scale
        if self.residual_scale_end is not None:
            # One ``t`` for the whole batch. The solver steps the batch on a
            # shared grid, and the caller that does not is the training path,
            # where this dial is off.
            t_frac = float(t_arg.reshape(-1)[0])
            gain = gain * (1.0 - t_frac) + self.residual_scale_end * t_frac
        if self.residual_scale_hi is not None:
            split = self.residual_scale_split_hz
            if split <= 0.0:
                # An octave above the cutoff: the default split puts the
                # boundary where the profile says the level is short and the
                # identity evidence has run out. ``cutoff_hz`` is per
                # utterance, so this is too.
                # ``cutoff_hz`` is only filled in above when the rate encoder
                # is built, so it can still be None here.
                hz = (
                    torch.full((batch,), self.cutoff_max, device=device)
                    if cutoff_hz is None
                    else self._expand_to_batch(cutoff_hz, batch)
                )
                split = 2.0 * hz.reshape(batch, 1, 1)
            centers = self.mel_centers.reshape(1, 1, -1)
            above = (centers > split).to(residual.dtype)
            # The dose the band needs is not one number. Measured per octave,
            # the same gain moves the top of the spectrum far more than the
            # octave above the cutoff: 2.5x over 4-6 kHz buys 9.3 dB there,
            # 1.2 times its nominal 8.0, while 2.0x over 8-12 kHz buys 16.3,
            # 2.7 times its nominal 6.0. Correcting an even 9 dB deficit
            # therefore wants a gain that falls with frequency, near 2.9x at
            # 3 kHz and 1.5x at 12 kHz, and no single step can express that.
            # So the raised dose interpolates log-linearly in frequency, from
            # ``residual_scale_hi`` at the split to ``residual_scale_top`` at
            # ``residual_scale_top_hz``, and is held flat beyond. Log rather
            # than linear because the deficit is flat per octave, not per Hz.
            # Leaving ``residual_scale_top`` unset makes the two ends equal,
            # which is the plain step every measurement before this used.
            top = (
                self.residual_scale_hi
                if self.residual_scale_top is None
                else self.residual_scale_top
            )
            top_hz = self.residual_scale_top_hz
            if top_hz <= 0.0:
                top_hz = float(self.mel_centers[-1])
            # ``split`` may be a per-utterance tensor, so this stays in tensor
            # arithmetic throughout and clamps rather than branching.
            span = (torch.log(torch.as_tensor(top_hz, device=device)) -
                    torch.log(torch.as_tensor(split, device=device)))
            frac = (torch.log(centers.clamp(min=1.0)) - torch.log(
                torch.as_tensor(split, device=device)
            )) / span.clamp(min=1e-3)
            frac = frac.clamp(0.0, 1.0)
            hi_gain = self.residual_scale_hi + (top - self.residual_scale_hi) * frac
            gain = gain * (1.0 - above) + hi_gain.to(residual.dtype) * above
        if not isinstance(gain, float) or gain != 1.0:
            residual = residual * gain
        if self.decouple_residual:
            self._decouple_prev = (t_now, residual)
        # Telemetry only; read by the training loop in place of the gate value
        # the earlier model logged, so that "how hard is the branch pushing"
        # stays visible on the training curves.
        self._residual_rms = residual.detach().pow(2).mean().sqrt()
        # Kept for the drift term, which needs the two halves of the sum apart:
        # the residual with its graph intact, since the gradient it is there to
        # create flows back through it, and the backbone's own velocity to
        # difference against. Stashed rather than returned so the solver's call
        # signature stays a plain tensor. ``forward`` reads them immediately
        # after its own call and nothing else may.
        self._last_residual = residual
        self._last_backbone_vt = vt.detach()
        if not self.band_fill:
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
        masked_features: Optional[torch.Tensor] = None,
        speaker_embedder: Optional[Callable] = None,
        drift_dt: float = 0.0,
        drift_normalize: bool = False,
        rollout_steps: int = 0,
        rollout_k: int = -1,
        rollout_prob: float = 1.0,
        rollout_decouple: bool = False,
        band_fill: bool = False,
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
        # ``masked_features`` is the degraded mel when the trainer has produced
        # one by really resampling the waveform. That is what evaluation feeds
        # the model, and it differs from ``band_limit``'s mel-domain floor by
        # 8.7 log-units over the removed band, so when it is available it is
        # used and ``band_limit`` is called only for the mask. See
        # ``resample_band_limit``.
        if masked_features is None:
            masked, band_mask = self._band_limit(features, cutoff_hz, randomize=True)
        else:
            _, band_mask = self._band_limit(features, cutoff_hz, randomize=False)
            masked = masked_features

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

        prompt_mask = ~speech_condition_mask & ~padding_mask
        # The pretrained speaker encoder is owned by the training script, not by
        # this model, so it arrives as a callable rather than a submodule and
        # nothing about it enters the checkpoint. It is handed the *condition* --
        # band-limited, prompt frames only -- for the reasons its own docstring
        # sets out.
        speaker_embedding = None
        if self.pretrained_spk_proj is not None and speaker_embedder is not None:
            embedding, valid_flag = speaker_embedder(speech_condition, prompt_mask)
            speaker_embedding = torch.cat([embedding, valid_flag], dim=-1)

        # The state the branch is fitted on: the straight line from noise to
        # the target, which is the flow-matching path.
        #
        # A rollout arm was built and measured to replace this with a state the
        # model itself produces, on the theory that the target's own high band
        # sitting in ``x_t`` lets the branch copy rather than invent. It does
        # not hold: the copy is the flow-matching task definition, not a
        # pathology, and every substitute origin tried either restored it or
        # collapsed the written band (+1.9 dB over stock at 4 kHz against this
        # path's +11.6 dB). Removed.
        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        # Scheduled sampling: replace that state with one the model produced
        # itself, by running the solver from noise under ``no_grad``.
        #
        # This is the one thing the two-term objective cannot reach. Measured
        # per octave, the deployed branch writes its band 9-10 dB too quietly,
        # and ``band_bias`` says that shortfall does not exist one step from
        # clean features -- the signed tile offset there is slightly *positive*.
        # The cause is exposure: on the flow path ``x_t`` already carries the
        # target's high band scaled by ``t``, so the branch learns a small
        # correction on top of a nearly-correct state, and at inference it
        # applies that same small correction to a state carrying only what it
        # has itself written. On a rolled-out state the band really is short, so
        # there is a signed error to descend, which is exactly what the
        # single-step terms lacked.
        #
        # The rollout is detached. Nothing is backpropagated through the solve:
        # the ``k`` forwards only build the state, and the graph is the single
        # attached forward below, so the memory cost is one extra activation
        # peak rather than ``k`` of them. The branch is still in the loop for
        # every rollout step, conditioning on its own accumulated writes, so
        # this is not a decoupled trajectory.
        #
        # ``ut`` has to change with it. ``features - noise`` is the velocity of
        # the straight line through ``noise``, and a rolled-out state is not on
        # that line; the target that still reaches ``features`` at ``t = 1`` is
        # ``(features - xt) / (1 - t)``, which is also what keeps ``x1_hat``
        # below equal to ``features`` when the prediction is exact.
        # Decoupled full-solve training. The branch reads the backbone's
        # hidden state at every step and its writes never re-enter what the
        # backbone reads, so the low band of the sample is bit-identical to the
        # frozen model's and ``SIM-o`` is preserved by construction rather than
        # by a penalty. Measured, the inference-time version of this holds
        # identity almost flat in the dose -- 0.368 at 1.0, 0.354 at 2.0, where
        # the coupled branch is already down to 0.198 at 1.5.
        #
        # Two consequences make the full solve affordable here, where it is not
        # affordable coupled. The adapter cannot influence the trajectory, so
        # the stock states are constants: ``_last_backbone_vt`` is detached and
        # the backbone is frozen, so no graph is built for any of the
        # ``rollout_steps`` backbone forwards. Only the adapter forwards carry
        # one, and the adapter is 256-dim and two layers against a 512-dim
        # 16-layer backbone.
        #
        # The point of paying for it is the loss, not the state. Every term in
        # this study so far has scored ``x1_hat`` one step from ground-truth
        # features, where the 9-10 dB level deficit does not exist -- it is
        # manufactured by the 16-step solve. Here ``x1_hat`` is the sample the
        # solver actually returns, so the deficit is in the objective.
        #
        # ``fm_loss`` has no meaning in this mode: there is no single ``t`` at
        # which a velocity is being matched. The endpoint terms are the
        # objective, so ``vt`` and ``ut`` are zeroed and ``fm_loss`` falls out.
        x1_hat_solved = None
        if band_fill and rollout_steps > 0:
            # The branch stops predicting a velocity. It reads the frozen
            # backbone's hidden state at every solver step, carries its own
            # state across all of them, and emits the missing band directly as
            # a mel. Nothing it produces is integrated, and nothing re-enters
            # what the backbone reads, so there is no trajectory to compete
            # with and no amplitude to lose across the solve.
            #
            # That removes the whole dose apparatus. ``residual_scale`` and the
            # frequency tilt exist to undo integration shrinkage -- the branch
            # writes 0.121 rms on the training path and 0.068 on a state it
            # rolled into -- and a direct read-out has no such shrinkage to
            # undo.
            #
            # It also removes the exposure problem rather than mitigating it.
            # The states the branch reads here are the states the solver really
            # visits, because this *is* the solve; training and sampling are the
            # same computation.
            #
            # The cost is the opposite failure. A velocity field carried the
            # stochasticity that kept the high band from collapsing onto its
            # conditional average, and a direct predictor under L1 gives the
            # conditional median, which is flatter than any real high band. If
            # the profile comes out smooth and quiet with the loss still
            # falling, that is this term, and the answer is an adversarial or
            # stochastic read-out rather than more capacity.
            grid = get_time_steps(
                t_start=0.0,
                t_end=1.0,
                num_step=rollout_steps,
                t_shift=1.0,
                device=features.device,
            )
            x_stock = noise
            state = None
            for step in range(rollout_steps):
                self.forward_fm_decoder(
                    t=grid[step],
                    xt=x_stock,
                    text_condition=text_condition,
                    speech_condition=speech_condition,
                    padding_mask=padding_mask,
                    band_mask=band_mask,
                    cutoff_hz=cutoff_hz,
                    prompt_mask=prompt_mask,
                    speaker_embedding=speaker_embedding,
                    adapter_state=state,
                )
                state = self._last_adapter_state
                # Detached and frozen, so the trajectory carries no graph; only
                # the branch's own state does.
                x_stock = x_stock + self._last_backbone_vt * (
                    grid[step + 1] - grid[step]
                )
            # Added in the mel domain rather than replacing the backbone's own
            # high band. Replacement is what "fill in" means and the read-out
            # can express it -- by learning the negative of what is there -- but
            # ``out_proj`` starts at std 1e-3, so a replacement would start the
            # band at zero mel, which is nowhere near the floor, while an
            # addition starts it at exactly the frozen model's behaviour.
            x1_hat_solved = x_stock + self._last_residual
            t = grid[-1].reshape(1, 1, 1).expand(features.size(0), 1, 1)
            ut = torch.zeros_like(noise)

        if rollout_decouple and rollout_steps > 0:
            grid = get_time_steps(
                t_start=0.0,
                t_end=1.0,
                num_step=rollout_steps,
                t_shift=1.0,
                device=features.device,
            )
            x_stock = noise
            delta = torch.zeros_like(noise)
            for step in range(rollout_steps):
                dt = grid[step + 1] - grid[step]
                self.forward_fm_decoder(
                    t=grid[step],
                    xt=x_stock,
                    text_condition=text_condition,
                    speech_condition=speech_condition,
                    padding_mask=padding_mask,
                    band_mask=band_mask,
                    cutoff_hz=cutoff_hz,
                    prompt_mask=prompt_mask,
                    speaker_embedding=speaker_embedding,
                )
                x_stock = x_stock + self._last_backbone_vt * dt
                delta = delta + self._last_residual * dt
            x1_hat_solved = x_stock + delta
            t = grid[-1].reshape(1, 1, 1).expand(features.size(0), 1, 1)
            # ``ut`` is the straight-line velocity from ``noise``; with no ``t``
            # to match it at, leaving it non-zero would put a constant in
            # ``fm_loss`` that no gradient can move.
            ut = torch.zeros_like(noise)

        if x1_hat_solved is None and rollout_steps > 0 and torch.rand(()).item() < rollout_prob:
            with torch.no_grad():
                grid = get_time_steps(
                    t_start=0.0,
                    t_end=1.0,
                    num_step=rollout_steps,
                    t_shift=1.0,
                    device=features.device,
                )
                # One ``k`` for the whole batch, so the rollout stays batched.
                # ``t`` is then per-batch rather than per-utterance, which costs
                # diversity within a step and none across a run.
                k = (
                    rollout_k
                    if rollout_k > 0
                    else int(torch.randint(1, rollout_steps, ()).item())
                )
                k = min(k, rollout_steps - 1)  # t < 1, or ``ut`` divides by zero
                rolled = noise
                for step in range(k):
                    v = self.forward_fm_decoder(
                        t=grid[step],
                        xt=rolled,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                        band_mask=band_mask,
                        cutoff_hz=cutoff_hz,
                        prompt_mask=prompt_mask,
                        speaker_embedding=speaker_embedding,
                    )
                    rolled = rolled + v * (grid[step + 1] - grid[step])
                xt = rolled
                t = grid[k].reshape(1, 1, 1).expand(features.size(0), 1, 1)
            ut = (features - xt) / (1.0 - t)

        self._capture_rng = drift_dt > 0.0
        vt = torch.zeros_like(noise) if x1_hat_solved is not None else self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            band_mask=band_mask,
            cutoff_hz=cutoff_hz,
            prompt_mask=prompt_mask,
            speaker_embedding=speaker_embedding,
        )
        self._capture_rng = False

        valid = ~padding_mask
        span_mask = speech_condition_mask & valid
        fm_loss = torch.mean((vt[span_mask] - ut[span_mask]) ** 2)

        # Trajectory drift: what the branch's own writes do to the bins it is
        # not allowed to write.
        #
        # The residual is added to a *velocity*, so the solver integrates it
        # into the next step's ``x_t``, and the frozen backbone reads the whole
        # spectrum. Its response is not band-diagonal: the off-diagonal block
        # ``dB_low / dx_high`` is nonzero, so a residual confined to 6% of the
        # bins moves the backbone's prediction in the other 94%. That is the
        # channel the speaker similarity leaves by, and it is why the two
        # earlier coupling attempts could not work -- within one step the low
        # band of ``x1_hat`` comes from ``v_b``, which has no functional
        # dependence on the residual, so their gradients never reached the
        # damaged bins at all.
        #
        # Here the response is measured directly: step the state along the
        # residual, ``delta = r * dt``, run the backbone on the same draw
        # again, and charge the low-band part of the difference. The branch
        # cannot subtract that response -- ``_write_mask`` forbids writing there
        # -- but ``dL/dr`` is nonzero through ``delta``, so it can choose a high
        # band that provokes less of one. Many high bands fit one utterance;
        # the ones that leave the backbone's low-band prediction alone are the
        # ones consistent with the speech already generated. So this is a
        # consistency term, not a brake on drive. If it turns out to act as a
        # brake -- similarity up while written energy and naturalness fall back
        # onto the curve the inference dose knob already traces -- the mechanism
        # is wrong and the term should go.
        #
        # One extra backbone forward, no extra parameters, nothing kept in the
        # checkpoint. Charged over the generated span, matching ``fm_loss``.
        drift_loss = torch.zeros((), device=vt.device, dtype=vt.dtype)
        if drift_dt > 0.0 and band_mask.any():
            delta = self._last_residual * drift_dt
            base_pert = torch.cat(
                [xt + delta, text_condition, speech_condition], dim=2
            )
            # Replay the draw the call above made, so dropout and stochastic
            # depth take the same path through the backbone in both forwards and
            # the difference is the perturbation alone. Without this the term
            # measures the backbone's own sampling noise, which is larger than
            # the signal and unrelated to the residual, so it has no useful
            # gradient and a weight on it is a weight on noise.
            cpu_rng, cuda_rng = self._rng_state
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, xt.device)
            vt_pert, _ = self.fm_decoder(
                x=base_pert,
                t=t.reshape(-1),
                padding_mask=padding_mask,
                return_hidden=True,
            )
            # Divided by the probe size, so what is charged is the directional
            # derivative ``J_low . r`` rather than ``J_low . r * dt``. Two
            # reasons: the term lands in velocity units, the same units
            # ``fm_loss`` is in, so its weight is readable against that rather
            # than against an arbitrary rescale; and ``drift_dt`` goes back to
            # being only a finite-difference step size, which is what it is.
            response = (vt_pert - self._last_backbone_vt) / drift_dt
            drift_weight = span_mask.unsqueeze(-1).to(vt.dtype) * (1.0 - band_mask)
            drift_loss = (response.pow(2) * drift_weight).sum() / drift_weight.sum(
            ).clamp(min=1.0)
            if drift_normalize:
                # ``|J.r|^2`` falls if the branch aims better *or* if it simply
                # writes less, and the second is the cheaper move: the term is
                # quadratic in ``r`` while everything that wants ``r`` large is
                # not. Measured against the branch trained without it, at
                # matched steps, drive ran 2.1-2.3x lower -- 0.0294 against
                # 0.0690 at step 3000 -- and band energy came out +0.76 dB over
                # the frozen model where the same branch untermed reaches
                # +11.6 dB. The term was braking, not steering.
                #
                # Dividing by the drive removes that move. ``|J.r|^2 / |r|^2``
                # is the gain along ``r``, unchanged when ``r`` is scaled, so
                # shrinking the residual buys nothing and only its *direction*
                # is charged.
                #
                # The denominator stays attached. That is the form that is
                # actually scale-invariant: with ``L = |J.r|^2 / |r|^2`` the
                # radial component of the gradient is
                # ``r . grad L = (2/|r|^2)(|J.r|^2 - L|r|^2) = 0`` exactly, so
                # descent can rotate ``r`` but cannot push its length either
                # way, which is the whole point. Detaching the denominator
                # breaks that: it leaves ``grad L = (2/c) J^T J r``, which
                # points along ``+r``, so descent shrinks the residual and the
                # term is a brake again -- now with a per-step adaptive weight,
                # which is worse than the unnormalised form, not better.
                #
                # The floor is what keeps the attached form from diverging.
                # ``1/|r|^2`` blows up if the branch collapses, so below a drive
                # of 1e-4 -- residual RMS 0.01, an order under anything a
                # working branch has shown (0.024 at the first validation of the
                # unnormalised arm, 0.14 at 30k untermed) -- the division is
                # frozen and the term degrades to the plain unnormalised drift
                # penalty rather than to a singularity.
                drive_weight = span_mask.unsqueeze(-1).to(vt.dtype) * band_mask
                drive = (self._last_residual.pow(2) * drive_weight).sum() / (
                    drive_weight.sum().clamp(min=1.0)
                )
                drift_loss = drift_loss / drive.clamp(min=1e-4)

        # Spectrogram reconstruction on the endpoint estimate, over the bins the
        # branch may write. ``band_mask`` is the soft ramp ``band_limit``
        # returns, so a bin on the transition slope is weighted by how much of it
        # was actually removed rather than counted as wholly present or wholly
        # absent, and normalising by the total weight makes a narrow cutoff and a
        # wide one contribute equally instead of letting 4 kHz dominate the
        # gradient simply by having more bins to invent.
        x1_hat = x1_hat_solved if x1_hat_solved is not None else xt + (1.0 - t) * vt
        weight = valid.unsqueeze(-1).to(vt.dtype) * band_mask  # (B, T, F)
        denom = weight.sum().clamp(min=1.0)
        recon_loss = ((x1_hat - features).abs() * weight).sum() / denom
        recon_loss = recon_loss * recon_loss_weight

        # Multi-resolution level match on the written band.
        #
        # ``recon_loss`` charges the error bin by bin and frame by frame, and
        # over the invented band most of that error is genuinely unpredictable:
        # the fine structure of a high band is not a function of the low band.
        # An L1 whose per-element optimum is the conditional median therefore
        # settles on a shrunken answer, and the shrinkage shows up as level.
        # Measured per octave against ground truth at a 4 kHz cutoff, the
        # coupled branch writes the right *shape* 9-10 dB too quietly:
        # -29.0 / -27.0 / -29.5 / -32.3 dB relative to its own low band across
        # 3-4k / 4-6k / 6-8k / 8-12k, where the references sit at
        # -19.9 / -18.0 / -19.9 / -22.6. Nothing in the two-term objective asks
        # for that level, so nothing supplies it.
        #
        # This term asks for it directly, and only for it. The error is pooled
        # over a tile of frames and bins *before* the absolute value is taken,
        # so deviations that cancel inside a tile cost nothing and only the net
        # offset of the tile is charged. That is the part of the band the low
        # band does predict. Several tile sizes are used because the level is
        # wrong at every scale and each one leaves a different residue: the
        # smallest still tracks a syllable, the largest is close to the
        # utterance's average spectrum.
        #
        # Always computed, never gated on its weight: three pooling passes are
        # nothing next to a backbone forward, and a run at weight 0 that logs it
        # says how far the level is off before any run pays to correct it.
        band_loss = torch.zeros((), device=vt.device, dtype=vt.dtype)
        band_bias = torch.zeros((), device=vt.device, dtype=vt.dtype)
        err = ((x1_hat - features) * weight).unsqueeze(1)  # (B, 1, T, F)
        w = weight.unsqueeze(1)
        for wt, wf in _BAND_LOSS_TILES:
            # ``count_include_pad`` is irrelevant here because numerator and
            # denominator are pooled identically and any common divisor cancels;
            # ``ceil_mode`` keeps the ragged last tile rather than dropping the
            # tail of every utterance.
            num = F.avg_pool2d(err, (wt, wf), ceil_mode=True)
            den = F.avg_pool2d(w, (wt, wf), ceil_mode=True)
            # A tile has to be mostly real to be scored. The threshold is on the
            # *mean* weight in the tile, so a tile that is half padding or half
            # below the cutoff is dropped rather than contributing a level
            # estimate built from a handful of elements.
            keep = (den > 0.25).to(vt.dtype)
            offset = num / den.clamp(min=1e-4)
            band_loss = band_loss + (offset.abs() * keep).sum() / keep.sum(
            ).clamp(min=1.0)
            # The same offsets without the absolute value. ``band_loss`` cannot
            # tell a branch that is systematically quiet from one whose tiles
            # are off in both directions by the same amount, and only the first
            # is correctable -- a zero-mean offset is the part of the band that
            # is not a function of the low band, and no weight on this term
            # recovers information that is not there. If this sits near zero
            # while ``band_loss`` does not, the term is charging noise.
            band_bias = band_bias + (offset * keep).sum() / keep.sum().clamp(
                min=1.0
            )
        band_loss = band_loss / len(_BAND_LOSS_TILES)
        band_bias = band_bias / len(_BAND_LOSS_TILES)

        # How much the written band varies *between* utterances. Both terms
        # above are minimised by a conditional mean: flow matching scores a
        # velocity and the reconstruction an absolute level, and the average
        # high band over every speaker in the batch does well on each. That is
        # the collapse the pitch-conditioned run measured -- naturalness held
        # while speaker similarity fell, which is what writing everyone's
        # average band looks like. Nothing in the objective noticed, because
        # collapse is not a worse solution by the objective.
        #
        # So the batch's own dispersion is made a target. Each utterance is
        # pooled to one spectrum over the frames and bins the branch may write,
        # the standard deviation across the batch is taken per bin, and the
        # prediction is charged for falling below the target's. The hinge is
        # deliberate: a branch that writes *more* varied bands than the data is
        # not collapsing, and this term has no business pushing it back. It is
        # a floor on diversity, not a distribution match.
        variance_loss = torch.zeros((), device=vt.device, dtype=vt.dtype)
        if features.size(0) > 1:
            pooled = weight.sum(dim=1)  # (B, F)
            keep_bin = pooled > 1.0
            pred_mean = (x1_hat * weight).sum(dim=1) / pooled.clamp(min=1.0)
            true_mean = (features * weight).sum(dim=1) / pooled.clamp(min=1.0)
            # Unbiased=False: with a handful of utterances per batch the n-1
            # correction is noise, and the two standard deviations are compared
            # with each other rather than read as absolute quantities.
            #
            # The dispersion at a bin is taken over the utterances that actually
            # write it, not over the batch. A bin an utterance does not write
            # contributes an exact zero to both means, which is dispersion coming
            # from the cutoff draw rather than from the speakers, so those rows
            # are excluded -- but excluding the *bin* whenever any row skips it,
            # which is what reducing with ``all`` does, throws away almost the
            # whole band: cutoffs are drawn per utterance, so the bins every row
            # writes are only those above the batch's widest draw, and a single
            # full-band utterance (``--full-band-prob``) writes nothing and
            # empties the term for the entire batch.
            count = keep_bin.sum(dim=0)  # (F,) utterances writing each bin
            k = keep_bin.to(vt.dtype)
            norm = count.clamp(min=1).to(vt.dtype)

            def masked_std(x: torch.Tensor) -> torch.Tensor:
                mean = (x * k).sum(dim=0) / norm
                var = (((x - mean) * k) ** 2).sum(dim=0) / norm
                # The epsilon is load-bearing, not cosmetic. A bin with fewer
                # than two writers has variance exactly zero, where sqrt has an
                # infinite derivative; the ``common`` mask below zeroes those
                # bins only *after* the square root, and zero times a NaN is a
                # NaN that reaches every branch tensor.
                return (var + 1e-8).sqrt()

            pred_std = masked_std(pred_mean)
            true_std = masked_std(true_mean)
            # Two writers minimum: a bin one utterance writes has no dispersion
            # to speak of, only its own deviation from itself, which is zero.
            common = (count >= 2).to(vt.dtype)
            shortfall = F.relu(true_std - pred_std) * common
            variance_loss = shortfall.sum() / common.sum().clamp(min=1)

        # ``condition_time_mask`` masks one contiguous window per utterance, so
        # the span is recovered from the mask itself rather than plumbed out of
        # it: the first masked frame, and that plus the count.
        span_len = speech_condition_mask.sum(dim=1)
        starts = speech_condition_mask.float().argmax(dim=1)
        aux = {
            "residual_rms": self._residual_rms,
            "variance_loss": variance_loss,
            "drift_loss": drift_loss,
            "band_loss": band_loss,
            "band_bias": band_bias,
            "x1_hat": x1_hat,
            "features": features,
            # What a discriminator over the written band has to condition on:
            # which bins were invented, how far the extrapolation reached, and
            # which frames are padding. All three are already computed above;
            # recomputing them in the trainer would duplicate the band-limiting
            # draw, which is random per step and would not agree.
            "band_mask": band_mask,
            "cutoff_hz": cutoff_hz,
            "frame_mask": valid.unsqueeze(-1).to(vt.dtype),
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
        for stream in (
            self.speaker_encoder,
            self.rate_encoder,
            self.pretrained_spk_proj,
        ):
            if stream is not None:
                stream.requires_grad_(True)
        self.adapter.requires_grad_(True)
        if mode == "decoder":
            self.fm_decoder.requires_grad_(True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _projected_sample(
        self,
        x: torch.Tensor,
        band_mask: torch.Tensor,
        num_step: int,
        guidance_scale: float,
        t_shift: float,
        **kwargs,
    ) -> torch.Tensor:
        """Euler solve that pins the low band to the frozen model's own solve.

        Two trajectories are integrated from the *same* noise on the *same*
        timestep grid: a reference one with the branch inert, and the adapter's.
        After each Euler step the adapter state takes the reference's low band
        back:

            x_adapter <- M_L * x_reference + M_H * x_adapter

        with ``M_H`` the write mask, so the blend boundary is exactly where the
        branch is allowed to write and the ramp is shared. The constraint is not
        the prompt -- the prompt is a different utterance and has no low band for
        *this* text. It is the low band the frozen backbone would itself have
        generated here, which is the only trusted reference this task has.

        The reference call passes an all-zero ``band_mask``: the residual is
        multiplied by ``_write_mask`` of it, so the branch contributes exactly
        zero and the velocity is the backbone's, identical to the dose-0 arm.

        ``project_trace`` records, per step, the low- and high-band norms of the
        divergence measured *before* the projection, which is where the damage
        accumulates if the state coupling is what causes it.
        """
        assert not self.decouple_residual, (
            "projection and decoupling are two answers to the same question; "
            "running both makes the result uninterpretable"
        )
        assert not self.band_fill, "band-fill writes after the solve, not during"

        timesteps = get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=num_step,
            t_shift=t_shift,
            device=x.device,
        )
        write = self._write_mask(band_mask)  # (B, 1, F), 1 where branch may write
        keep = 1.0 - write
        zero_mask = torch.zeros_like(band_mask)

        x_ref = x
        x_adp = x
        trace = []
        for step in range(num_step):
            dt = timesteps[step + 1] - timesteps[step]
            v_ref = self.solver.model(
                t=timesteps[step],
                x=x_ref,
                guidance_scale=guidance_scale,
                band_mask=zero_mask,
                **kwargs,
            )
            v_adp = self.solver.model(
                t=timesteps[step],
                x=x_adp,
                guidance_scale=guidance_scale,
                band_mask=band_mask,
                **kwargs,
            )
            x_ref = x_ref + v_ref * dt
            x_adp = x_adp + v_adp * dt
            diff = x_adp - x_ref
            trace.append(
                (
                    step,
                    float(timesteps[step + 1]),
                    float((diff * keep).pow(2).sum().sqrt()),
                    float((diff * write).pow(2).sum().sqrt()),
                )
            )
            x_adp = keep * x_ref + write * x_adp
        self.project_trace = trace
        return x_adp

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
        speaker_embedding: Optional[torch.Tensor] = None,
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

        # Per-utterance-batch state: the accumulated branch contribution and the
        # previous step's residual, both meaningless across calls.
        self._decouple_delta = None
        self._decouple_prev = None
        self._bf_state = None

        solve = (
            self._projected_sample if self.project_low_band else self.solver.sample
        )
        x1 = solve(
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
            speaker_embedding=speaker_embedding,
        )

        if self.band_fill:
            # The branch predicted a band, not a velocity, so it joins the mel
            # once at the end rather than the velocity at every step.
            band = self._last_residual
            if band.size(0) != x1.size(0):
                # Classifier-free guidance ran the step at twice the batch, the
                # conditional half second. The branch's output is a band, not a
                # velocity, so extrapolating it away from an unconditional band
                # has no meaning; take the conditional one.
                band = band.chunk(2, dim=0)[1]
            x1 = x1 + band

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
