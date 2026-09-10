# Copyright    2025    Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../../LICENSE for clarification regarding multiple authors
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

"""ZipVoice with frequency-aware conditioning, after AnyBand.

Motivation
----------

A zero-shot text-to-speech model copies the spectral character of its prompt,
because that is what in-context conditioning is for. When the prompt is a
band-limited recording -- a telephone call, a compressed clip -- the copied
character includes the missing high band, and the model dutifully synthesises
speech with the same dead upper region even though its output is nominally
24 kHz. Measurement in this repository confirmed this directly: the output's
cutoff tracks the prompt's cutoff exactly. The problem is therefore not missing
information but a copy prior, and conditioning that merely *informs* the model
about bandwidth is not enough. It has to be able to override the prior.

Two mechanisms give it that ability here.

The first is a conditioning path built the way AnyBand builds one: attention
along the frequency axis, so the model reasons about the relation between mel
bins rather than seeing each frame as an unstructured vector, plus an explicit
F0 and voicing contour. See :mod:`zipvoice.models.modules.anyband`.

The second is the schedule. Flow matching settles coarse, semantic structure in
its early steps and detail in its late ones, so the two conditioning sources are
wanted at different points along the trajectory: text and the prompt mel early,
when what is being decided is what is said and by whom, and the frequency path
late, when what is being decided is the spectral fine structure the prompt
cannot supply. A single monotone gate ``alpha(t)`` interpolates between them,
weighting the base conditioning ``1 - alpha`` and the frequency conditioning
``alpha``. It is learned, not fixed, so the crossover is found rather than
guessed; it is monotone by construction, so it can only ever move in the
intended direction; and it is one gate rather than two, so the model cannot
quietly switch the new path off and keep the old one.

Preserving the pretrained model
-------------------------------

The backbone here is a released ZipVoice checkpoint and the intent is to add to
it, not to retrain it. That constraint is sharper than it looks, because of a
result measured in this repository on an earlier conditioning experiment: the
optimiser used for training, ScaledAdam, factors each update into a direction
and a *per-tensor scale* that multiplies the whole parameter. Freezing part of a
tensor by zeroing part of its gradient therefore does not work -- the released
columns of a widened ``in_proj`` came out uniformly rescaled by 0.904 after
30000 iterations, despite their gradient being zeroed throughout.

The conclusion is that no pretrained tensor may be widened. Every new path
instead enters through its own bias-free, zero-initialised ``Linear`` whose
output is *added* to the existing projection's output. Concatenating two blocks
and multiplying by one matrix is arithmetically identical to multiplying each by
its own matrix and summing, so this is the same function -- but the pretrained
tensor stays a separate parameter that the optimiser never has to touch.

Combined with the gate's warm-up, which scales ``alpha`` in from zero over the
first few thousand steps, the model at step 0 is the released checkpoint exactly
and departs from it only as fast as it learns something.
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipvoice.models.modules.anyband import (
    F0Embedding,
    FrequencyDecoder,
    FrequencyEncoder,
    MelF0Estimator,
    TimestepGate,
)
from zipvoice.models.modules.bwe import (
    LOG_MEL_FLOOR,
    band_limit,
    mel_bin_centers,
    sample_cutoff,
)
from zipvoice.models.modules.solver import EulerSolver
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, make_pad_mask


class ZipVoiceAnyBand(ZipVoice):
    """ZipVoice with an AnyBand-style frequency conditioning path.

    Subclasses rather than edits :class:`~zipvoice.models.zipvoice.ZipVoice`,
    following the precedent of the other variants in this package: the
    distillation, dialogue and ONNX/TensorRT export paths all depend on the base
    class staying where it is.
    """

    def __init__(
        self,
        freq_dim: int = 64,
        freq_num_layers: int = 2,
        freq_num_heads: int = 4,
        freq_bin_out: int = 4,
        f0_hidden_dim: int = 64,
        gate_num_knots: int = 8,
        gate_warmup_steps: int = 2000,
        gate_max_alpha: float = 0.98,
        gate_fixed_schedule: bool = False,
        gate_base: bool = False,
        f0_extract_t: float = 0.5,
        f0_estimator_channels: int = 192,
        use_f0: bool = False,
        ltas_channels: int = 4,
        use_freq_decoder: bool = True,
        branch_entry: str = "input",
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Degradation settings, shared verbatim with the bandwidth-extension
        # baseline so the two methods see the same simulated band-limiting.
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
        # A fraction of every batch is left full-band. The curriculum alone is
        # continuous on [cutoff_min, cutoff_max] and so never produces the
        # unmasked condition the backbone was pretrained on and every 24 kHz
        # evaluation runs at; without this the gate only ever sees
        # ``band_frac > 0`` in training and has to extrapolate at test time.
        self.full_band_prob = full_band_prob

        decoder_dim = self.fm_decoder.in_proj.out_features
        self.freq_encoder = FrequencyEncoder(
            num_bins=self.feat_dim,
            dim=freq_dim,
            num_layers=freq_num_layers,
            num_heads=freq_num_heads,
            bin_out=freq_bin_out,
            out_dim=decoder_dim,
            ltas_channels=ltas_channels,
        )
        # F0 and voicing are off by default. They were the paper's own inputs,
        # but the paper extracts them from the band-limited *waveform* it is
        # restoring, which exists over the whole span. Here there is no waveform
        # over the span being generated, so training fed the estimator the clean
        # ground-truth mel of the entire utterance while inference fed it a
        # one-shot endpoint guess. Voicing is phonemic and per frame, so what
        # training supplied over the generated span was a phonetic skeleton of
        # the answer. The gate learned to trust it -- alpha settled at 0.94
        # irrespective of bandwidth -- and at inference the contour is instead a
        # single Euler extrapolation from a band-limited prompt. Word error rose
        # with bandwidth, 1.6x stock at 4 kHz and 7.7x at 16 kHz, precisely the
        # ordering a confidently wrong contour predicts. Kept behind a flag
        # rather than deleted so the ablation remains runnable.
        self.use_f0 = use_f0
        self.f0_embed = (
            F0Embedding(out_dim=decoder_dim, hidden_dim=f0_hidden_dim)
            if use_f0
            else None
        )

        # The new path enters the backbone by having its output added to
        # ``in_proj``'s, which is what keeps the released ``in_proj`` out of the
        # optimiser entirely; see the module docstring on why widening
        # ``in_proj`` instead is not an option. That entry point is the
        # frequency encoder's own ``out_proj``. An earlier version put a second
        # bias-free ``Linear(decoder_dim, decoder_dim)`` after it, zero
        # initialised so the model was bit-identical to the released one at step
        # 0. Both properties of that layer were mistakes:
        #
        #   * Two linear layers in sequence with nothing between them compose to
        #     one linear layer, so it added no capacity, only a worse-conditioned
        #     factorisation of the same map.
        #   * Zero initialisation is fatal under ScaledAdam. Every direction step
        #     is scaled by the tensor's own per-row rms clamped below at
        #     ``param_min_rms`` = 1e-5, and the multiplicative scale update --
        #     the only mechanism that could grow the tensor -- is explicitly
        #     suppressed for any row beneath that floor. A row starting at
        #     exactly zero can therefore never escape. Measured on the first run
        #     of this model: rms 6.15e-06 at 1k iterations, 5.95e-06 at 9k,
        #     shrinking under weight decay rather than learning, with the
        #     frequency encoder and F0 embedding decaying alongside it because
        #     their gradient arrives through this matrix. The whole branch was
        #     inert for the entire run while the gate still charged the base
        #     conditioning ``1 - alpha`` for it.
        #
        # The step-0 identity property is supplied by the gate's warm-up, which
        # scales alpha -- and with it this branch's entire contribution -- in
        # from zero over ``gate_warmup_steps``. It never needed a zero-init
        # matrix as well.

        self.freq_decoder = (
            FrequencyDecoder(
                num_bins=self.feat_dim,
                in_dim=decoder_dim,
                dim=freq_dim,
                num_layers=freq_num_layers,
                num_heads=freq_num_heads,
            )
            if use_freq_decoder
            else None
        )

        # Two gates, because the branch enters the frozen backbone at two places
        # whose units have nothing to do with each other. ``gate_enc`` scales
        # the encoder's term into the residual stream after ``in_proj``, where a
        # coordinate is a backbone channel and a per-bin quantity would be
        # meaningless, so it stays the scalar alpha. ``gate_dec`` scales the
        # read-out, whose contribution is one velocity per mel bin, so it is
        # per bin. Sharing one scalar across both made every gain above the
        # cutoff cost a disturbance below it, which is what the speaker
        # similarity regression at full band was reading.
        #
        # Both read the prompt's long-term average spectrum alongside the band
        # fraction, so how far the branch opens is a question about this voice
        # and this recording rather than about bandwidth alone.
        gate_kwargs = dict(
            num_knots=gate_num_knots,
            warmup_steps=gate_warmup_steps,
            max_alpha=gate_max_alpha,
            cond_dim=self.feat_dim if ltas_channels else 0,
        )
        self.gate_enc = TimestepGate(**gate_kwargs)
        self.gate_dec = TimestepGate(out_bins=self.feat_dim, **gate_kwargs)
        # The frozen-gate ablation: alpha follows the hand-designed 8/4/4 ramp of
        # a 16-step solver and never adapts, isolating what the learned schedule
        # is worth against a reasonable fixed one.
        self.gate_fixed_schedule = gate_fixed_schedule
        self.gate_base = gate_base

        # Where the branch reaches the frozen backbone.
        #
        # ``input`` sums the frequency encoder's output into the residual stream
        # after ``in_proj``, which is what the first version did throughout.
        # Measured against a four-arm inference ablation, that entry point
        # carries every bit of the branch's effect -- muting it at 4 kHz costs
        # 0.60 UTMOS, muting the read-out instead changes nothing at all -- and
        # every bit of its damage: at full band, where ``band_mask`` is all zero
        # and the branch has nothing to contribute, it still costs 0.066 speaker
        # similarity and 1.7 points of word error against the released
        # checkpoint. The mechanism is the obvious one. A vector summed into the
        # residual stream moves the point at which the frozen backbone evaluates
        # its own vector field, and the backbone was never trained at the moved
        # point. A scalar gate cannot buy the first effect without paying the
        # second, because "off at full band" and "on for these channels" are not
        # separable by one number.
        #
        # ``output`` routes the same encoder output into the read-out head
        # instead, so the residual stream is never touched and the branch can
        # only add to the predicted velocity. The frozen backbone then runs at
        # exactly the point it was trained on for any bandwidth, which is what
        # makes content robustness structural here rather than something the
        # gate has to learn. The read-out's per-bin gate already conditions on
        # the band fraction, so switching the branch off at full band is within
        # its reach in a way it was not for the scalar.
        #
        # Note this is not the same as the ``use_freq_decoder=False`` ablation
        # read in reverse: with ``output`` the frequency encoder keeps all 312k
        # of its parameters and its gradient, it simply arrives by a different
        # route. Zeroing ``alpha_enc`` alone would instead strand the encoder
        # with no gradient at all and test the idea at a third of its capacity.
        if branch_entry not in ("input", "output"):
            raise ValueError(
                f"branch_entry must be input or output, got {branch_entry}"
            )
        if branch_entry == "output" and self.freq_decoder is None:
            raise ValueError(f"branch_entry={branch_entry!r} needs use_freq_decoder")
        self.branch_entry = branch_entry
        if branch_entry == "output":
            # ``alpha_enc`` multiplies nothing in these modes. Freezing it keeps
            # parameters that cannot receive gradient out of the optimiser.
            for p in self.gate_enc.parameters():
                p.requires_grad_(False)

        if gate_fixed_schedule:
            for g in (self.gate_enc, self.gate_dec):
                for p in g.parameters():
                    p.requires_grad_(False)

        # Stands in for pYIN, which needs a waveform the model does not have over
        # the span it is generating. Trained separately and frozen; see
        # ``zipvoice/bin/train_f0_estimator.py``.
        self.f0_estimator = (
            MelF0Estimator(num_bins=self.feat_dim, channels=f0_estimator_channels)
            if use_f0
            else None
        )
        if self.f0_estimator is not None:
            self.f0_estimator.requires_grad_(False)
        self.f0_extract_t = f0_extract_t

        # The base class points the solver at ``forward_fm_decoder``. Redirect
        # it to the wrapper that owns the one-off F0 extraction, so that every
        # caller of ``sample`` gets it without having to know about it.
        self.solver = EulerSolver(self, func_name="_fm_decoder_with_f0")

        # Filled in by :meth:`sample` at the first ODE step past
        # ``f0_extract_t`` and reused for the rest of that trajectory.
        self._f0_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def _ltas(
        condition: torch.Tensor, prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Average the observed condition over prompt frames, giving (B, F).

        ``prompt_mask`` is (B, T), true on the frames whose mel the model is
        actually given. Averaging anywhere else is what made the F0 path leak:
        a statistic pooled over the span being generated is available in
        training and not at inference. Every frame this reads is present and
        identical in both.
        """
        w = prompt_mask.to(condition.dtype).unsqueeze(-1)
        return (condition * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def _alphas(
        self,
        t: torch.Tensor,
        batch: int,
        device,
        band_mask: torch.Tensor,
        ltas: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both gates for a t of any of the shapes the solver may pass.

        ``band_mask`` is (B, 1, F) and marks the bins the prompt does not
        observe, so its mean is the fraction of the spectrum the frequency path
        has to invent. That is the bandwidth the gate conditions on: it is the
        same quantity a sampling rate would carry, but it is already batch
        expanded and already present at every training and inference call site,
        so it needs no second parallel path of its own.
        """
        flat = t.reshape(-1) if t.dim() > 0 else t.reshape(1)
        if flat.numel() == 1:
            flat = flat.repeat(batch)
        flat = flat.to(device)
        band_frac = band_mask.reshape(batch, -1).mean(dim=1)
        alpha_enc = self.gate_enc(flat, band_frac, ltas).reshape(batch, 1, 1)
        alpha_dec = self.gate_dec(flat, band_frac, ltas).reshape(
            batch, 1, self.feat_dim
        )
        # Inference-time ablation of either entry point, for attributing a
        # regression to one of them without retraining. The branch reaches the
        # frozen backbone in two places -- a vector summed into the residual
        # stream, and a per-bin residual on the velocity -- and zeroing one
        # alpha removes exactly one of them. With both zeroed the model has to
        # reproduce the released checkpoint bit for bit, which is what makes the
        # other two arms interpretable.
        ablate = os.environ.get("ANYBAND_ABLATE_GATE", "")
        if ablate in ("enc", "both"):
            alpha_enc = torch.zeros_like(alpha_enc)
        if ablate in ("dec", "both"):
            alpha_dec = torch.zeros_like(alpha_dec)
        return alpha_enc, alpha_dec

    @staticmethod
    def _expand_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
        """Repeat a tensor along dim 0 to match a batch grown by guidance."""
        if x.size(0) == batch:
            return x
        assert batch % x.size(0) == 0, (batch, x.shape)
        return x.repeat(batch // x.size(0), *([1] * (x.dim() - 1)))

    def _estimate_f0(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen estimator and return ``(f0_norm, voiced)``."""
        with torch.no_grad():
            f0_norm, voiced_logit = self.f0_estimator(mel)
        return f0_norm, (voiced_logit > 0).to(mel.dtype)

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        band_mask: Optional[torch.Tensor] = None,
        f0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity, blending the base and frequency conditioning.

        Args:
            band_mask: True where a mel bin is missing from
                ``speech_condition``, of shape (B, 1, F). When None the
                condition is treated as full band, which makes the frequency
                encoder's observed-bin flag uniformly true.
            f0: an optional ``(f0_norm, voiced)`` pair, each of shape (B, T).
                When None the F0 embedding is skipped, which is the correct
                behaviour early in the trajectory, before an endpoint estimate
                good enough to extract a contour from exists.

        The other arguments are as in the base class.
        """
        batch, num_frames, _ = xt.shape
        device = xt.device

        if band_mask is None:
            band_mask = torch.zeros(batch, 1, self.feat_dim, device=device)
        # Classifier-free guidance stacks the unconditional and conditional
        # halves into one batch inside the solver, but only touches the
        # arguments it knows about. The extra conditioning has to follow, and it
        # is the same for both halves: what guidance varies is the condition
        # content, not which bins the prompt happened to observe.
        band_mask = self._expand_to_batch(band_mask, batch)
        if ltas is not None:
            ltas = self._expand_to_batch(ltas, batch)
        if f0 is not None:
            f0 = (
                self._expand_to_batch(f0[0], batch),
                self._expand_to_batch(f0[1], batch),
            )

        alpha, alpha_dec = self._alphas(t, batch, device, band_mask, ltas)

        # ``xt`` is the integration state rather than conditioning, so it is
        # never gated.
        #
        # Whether the *base* conditioning is weighted down by ``1 - alpha`` as
        # the frequency path comes up is ``gate_base``, and it defaults off.
        # The original argument for gating it was that a single tied alpha makes
        # ignoring the new path cost the model its old one, ruling out the "base
        # on, frequency off" collapse an earlier experiment here fell into. What
        # that argument misses is that ``1 - alpha`` is a subtraction, not a
        # routing decision: it destroys text and prompt information the backbone
        # needs, and no additive frequency path can put it back. Measured on the
        # first run, that is exactly what happened -- the branch was inert (see
        # __init__ on the zero-init/ScaledAdam interaction) while alpha still
        # settled near 0.53, so the model paid away roughly half its
        # conditioning for a scalar, and ``fm_loss`` drifted upward over the
        # final third of training. The collapse is better prevented by making
        # the new path actually trainable than by holding the old one hostage.
        base_scale = (1 - alpha) if self.gate_base else 1.0
        base = torch.cat(
            [
                xt,
                text_condition * base_scale,
                speech_condition * base_scale,
            ],
            dim=2,
        )

        embedding = self.freq_encoder(xt, speech_condition, band_mask, ltas)
        if f0 is not None and self.f0_embed is not None:
            embedding = embedding + self.f0_embed(f0[0], f0[1])
        if self.branch_entry == "input":
            extra_in = embedding * alpha
            read_out_extra = None
        else:
            extra_in = None
            read_out_extra = embedding

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
        else:
            guidance_scale = None

        vt, hidden = self.fm_decoder(
            x=base,
            t=t_arg,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            extra_in=extra_in,
            return_hidden=True,
        )

        if self.freq_decoder is not None:
            # A second read-out that predicts each bin from a representation
            # which has attended to the others, on its own per-bin gate so that
            # opening the invented band above the cutoff does not oblige the
            # model to disturb the bins the prompt already gave it.
            vt = vt + self.freq_decoder(hidden, read_out_extra)[..., 0] * alpha_dec

        assert vt.shape[1] == num_frames, (vt.shape, num_frames)
        return vt

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
        infill_loss_weight: float = 1.0,
        var_loss_weight: float = 0.0,
        f0_noise_std: float = 0.05,
        need_adv: bool = False,
        adv_t_min: float = 0.5,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict]
    ]:
        """Forward pass for training.

        Two losses are returned. The first is ZipVoice's own flow-matching loss
        over the generated span, unchanged, which keeps the model a
        text-to-speech model. The second is the reason this class exists.

        AnyBand's loss is restricted to the bins missing from its input and
        normalised by their count, so the objective is entirely about infilling.
        Transplanting that naively into text-to-speech does nothing: over the
        generated span *no* bin is observed, so the restriction is vacuous and
        the loss collapses back to the ordinary one. The band-limited region that
        actually has an observed low band is the *prompt*, and the prompt is
        exactly what the ordinary loss excludes.

        So the second term puts it back: over the prompt frames, on the bins the
        band-limiting removed, normalised by their count. Those bins are missing
        from the condition but present in ``features``, which makes this dense,
        honest supervision on precisely the task -- predict a high band given a
        low band and a voice -- with real low-band context to predict it from. An
        earlier attempt in this repository band-limited the prompt and then
        excluded it from the loss, so the model was never once asked to do that,
        which is the most likely reason it learned nothing.
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

        # At inference the contour is extracted from the model's own endpoint
        # estimate once the trajectory has passed ``f0_extract_t``, and is
        # unavailable before that. Training mirrors both halves. The ideal
        # endpoint estimate is the clean mel, so that is what the estimator sees,
        # roughened with noise so the model does not come to rely on a contour
        # cleaner than the one it will actually be given.
        # Pooled over the prompt frames only, which are the frames whose mel
        # the model is given at inference too.
        ltas = self._ltas(speech_condition, ~speech_condition_mask & ~padding_mask)

        t_scalar = t.reshape(t.shape[0], -1)[:, 0]
        if self.use_f0 and bool((t_scalar >= self.f0_extract_t).any()):
            mel_for_f0 = features
            if f0_noise_std > 0:
                mel_for_f0 = mel_for_f0 + torch.randn_like(mel_for_f0) * f0_noise_std
            f0_norm, voiced = self._estimate_f0(mel_for_f0)
            keep = (t_scalar >= self.f0_extract_t).to(f0_norm.dtype).unsqueeze(-1)
            f0 = (f0_norm * keep, voiced * keep)
        else:
            f0 = None

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            band_mask=band_mask,
            f0=f0,
            ltas=ltas,
        )

        valid = ~padding_mask
        span_mask = speech_condition_mask & valid
        fm_loss = torch.mean((vt[span_mask] - ut[span_mask]) ** 2)

        # The infilling term: prompt frames, missing bins only, normalised by
        # their total weight so a narrow cutoff and a wide one contribute
        # equally. ``band_mask`` is the soft ramp ``band_limit`` returns, so a
        # bin on the transition slope is weighted by how much of it was actually
        # removed rather than being counted as wholly present or wholly absent.
        prompt_mask = ((~speech_condition_mask) & valid).unsqueeze(-1)  # (B, T, 1)
        weight = prompt_mask.to(vt.dtype) * band_mask  # (B, T, F)
        denom = weight.sum().clamp(min=1.0)
        infill_loss = ((vt - ut) ** 2 * weight).sum() / denom

        # A squared-error term on the missing bins is minimised by the
        # conditional mean, and the conditional mean of a high band is flatter
        # than any high band ever is. That is the standard over-smoothing of
        # regression-trained bandwidth extension, and it is audible as a
        # muffled, buzzy upper register -- exactly what a MOS predictor
        # penalises, and the largest part of the gap to the cascade at 4 kHz,
        # where most of the spectrum is invented.
        #
        # The fix is to ask for the right amount of variation as well as the
        # right mean. The endpoint estimate implied by the predicted velocity is
        # ``xt + (1 - t) * vt``; its per-bin standard deviation along time, over
        # prompt frames, is compared with the same statistic of the true mel and
        # the gap is charged on the missing bins. Matching a second moment says
        # nothing about where the detail goes, only that there has to be some,
        # which is all a squared-error term is structurally unable to ask for.
        var_loss = torch.zeros((), device=vt.device, dtype=vt.dtype)
        if var_loss_weight > 0:
            x1_hat = xt + (1.0 - t) * vt
            pw = prompt_mask.to(vt.dtype)
            n = pw.sum(dim=1).clamp(min=1.0)  # (B, 1)
            var_pred = (
                ((x1_hat - (x1_hat * pw).sum(1, keepdim=True) / n.unsqueeze(1)) ** 2)
                * pw
            ).sum(1) / n
            var_true = (
                ((features - (features * pw).sum(1, keepdim=True) / n.unsqueeze(1)) ** 2)
                * pw
            ).sum(1) / n
            bw = band_mask.reshape(band_mask.size(0), -1)  # (B, F)
            # The offset is not cosmetic: d/dx sqrt(x) is unbounded at zero and
            # a prompt of a few frames can put a bin's variance there exactly.
            eps = 1e-8
            var_loss = (
                (
                    (var_pred.clamp(min=0) + eps).sqrt()
                    - (var_true + eps).sqrt()
                )
                ** 2
                * bw
            ).sum() / bw.sum().clamp(min=1.0)
            infill_loss = infill_loss + var_loss * var_loss_weight

        infill_loss = infill_loss * infill_loss_weight
        var_loss = var_loss.detach()

        alpha_enc, alpha_dec = self._alphas(
            t, features.size(0), features.device, band_mask, ltas
        )
        alpha_mean = alpha_enc.mean().detach()

        adv_ctx = None
        if need_adv:
            adv_ctx = self._adv_context(
                features=features,
                xt=xt,
                vt=vt,
                t_scalar=t_scalar,
                span_mask=span_mask,
                band_mask=band_mask,
                adv_t_min=adv_t_min,
            )
        return fm_loss, infill_loss, var_loss, alpha_mean, adv_ctx

    @staticmethod
    def _adv_context(
        features: torch.Tensor,
        xt: torch.Tensor,
        vt: torch.Tensor,
        t_scalar: torch.Tensor,
        span_mask: torch.Tensor,
        band_mask: torch.Tensor,
        adv_t_min: float,
    ) -> Optional[dict]:
        """Assemble what a mel discriminator needs from one training step.

        The discriminator judges the *endpoint estimate*
        ``M_hat = x_t + (1 - t) * v_t`` rather than the velocity, because the
        endpoint estimate is a mel and the velocity is not: an adversary trained
        on velocities would be judging a quantity that never reaches a vocoder.

        Two restrictions make the comparison meaningful. Only the generated span
        is shown, since the prompt frames are discarded before vocoding and a
        discriminator that could see them would win by copying. And only samples
        whose timestep has passed ``adv_t_min`` take part, because at low ``t``
        the endpoint estimate is dominated by noise and separating it from a
        clean mel is trivial, which teaches the generator nothing.

        Returns ``None`` when no sample in the batch is late enough, which the
        caller treats as "skip the adversarial terms for this step".
        """
        select = t_scalar >= adv_t_min
        if not bool(select.any()):
            return None

        mel_hat = xt + (1.0 - t_scalar.reshape(-1, 1, 1)) * vt
        frame_mask = span_mask.unsqueeze(-1).to(mel_hat.dtype)
        return {
            "real": features[select],
            "fake": mel_hat[select],
            "frame_mask": frame_mask[select],
            "band_mask": band_mask[select],
        }

    def set_trainable(self, mode: str) -> None:
        """Select which parameters the optimiser sees.

        ``anyband`` trains only the new modules, leaving every released tensor
        untouched -- not merely un-updated but absent from the optimiser, which
        is what makes the result mergeable back onto the released checkpoint.
        ``decoder`` additionally releases the flow-matching decoder at a low
        learning rate, keeping the text encoder and token embedding fixed: they
        carry no bandwidth information, so there is nothing for the new path to
        teach them, and the released model was tuned around that text
        representation. ``all`` releases everything and exists to measure what
        the restraint costs.
        """
        assert mode in ("anyband", "decoder", "all"), mode
        new_modules = [
            self.freq_encoder,
            self.gate_enc,
            self.gate_dec,
        ]
        if self.f0_embed is not None:
            new_modules.append(self.f0_embed)
        if self.freq_decoder is not None:
            new_modules.append(self.freq_decoder)

        if mode == "all":
            self.requires_grad_(True)
        else:
            self.requires_grad_(False)
            for module in new_modules:
                module.requires_grad_(True)
            if mode == "decoder":
                self.fm_decoder.requires_grad_(True)

        if self.gate_fixed_schedule:
            self.gate_enc.requires_grad_(False)
            self.gate_dec.requires_grad_(False)
        if self.branch_entry == "output":
            # ``alpha_enc`` scales nothing in this mode, so its gradient is not
            # merely small but absent. Leaving it in the optimiser would hand
            # ScaledAdam a parameter with ``grad=None``.
            self.gate_enc.requires_grad_(False)
        # The estimator is a fixed feature extractor in every mode.
        if self.f0_estimator is not None:
            self.f0_estimator.requires_grad_(False)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _fm_decoder_with_f0(self, t, xt, **kwargs) -> torch.Tensor:
        """Solver entry point that manages the one-off F0 extraction.

        Before ``f0_extract_t`` there is no contour and none is passed. At the
        first step past it, the velocity is computed once without a contour, the
        endpoint estimate ``xt + (1 - t) * v`` is formed from it -- AnyBand's own
        construct -- the contour is extracted from that and cached, and the step
        is recomputed with it. Every later step reuses the cache, so the extra
        cost is a single duplicated step per utterance.
        """
        if not self.use_f0:
            return self.forward_fm_decoder(t=t, xt=xt, f0=None, **kwargs)

        t_value = float(t.reshape(-1)[0])

        if t_value < self.f0_extract_t:
            return self.forward_fm_decoder(t=t, xt=xt, f0=None, **kwargs)

        if self._f0_cache is None:
            v = self.forward_fm_decoder(t=t, xt=xt, f0=None, **kwargs)
            mel_hat = xt + (1.0 - t_value) * v
            self._f0_cache = self._estimate_f0(mel_hat)

        return self.forward_fm_decoder(t=t, xt=xt, f0=self._f0_cache, **kwargs)

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
        frequency encoder which bins of the prompt are genuinely observed as
        opposed to merely quiet; passing None declares the prompt full band.
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

        if cutoff_hz is None:
            band_mask = torch.zeros(batch_size, 1, self.feat_dim, device=device)
        else:
            # The prompt arrives already band-limited by its own recording rate,
            # so only the mask is wanted here; re-imposing the floor as well
            # would be a second degradation on top of the real one.
            _, band_mask = self._band_limit(
                speech_condition,
                torch.full((batch_size,), float(cutoff_hz), device=device),
                randomize=False,
            )

        # The same statistic the training pass pools, over the same frames:
        # those the prompt actually occupies.
        ltas = self._ltas(speech_condition, ~speech_condition_mask & ~padding_mask)

        x0 = torch.randn(
            batch_size, num_frames, prompt_features.size(-1), device=device
        )

        self._f0_cache = None
        try:
            x1 = self.solver.sample(
                x=x0,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                num_step=num_step,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                band_mask=band_mask,
                ltas=ltas,
            )
        finally:
            self._f0_cache = None

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
