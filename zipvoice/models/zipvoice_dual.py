# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
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

"""Two flow-matching models solved side by side, one per frequency band.

The band a narrowband prompt does not contain is not a function of the band it
does contain plus the text: a speaker's 8-12 kHz energy is one of many bands
consistent with everything the model is told. A regressor trained on that with
an L1 or L2 reconstruction term therefore converges on the per-bin conditional
median, which in log-mel sits 9-10 dB below the energy of any real sample. That
is what the earlier ControlNet-style adapter measured (see the ``adapter-arm``
branch), and no amount of tuning moves it, because the hedge is the objective's
optimum rather than a failure to reach it.

So the high band is generated here instead of predicted. :class:`HighBandBranch`
carries its own flow-matching state and its own velocity field, and a sample
from it is a draw from the conditional distribution rather than its centre. The
flow-matching term alone therefore has no reason to hedge.

The branch predicts the endpoint ``x1`` rather than the velocity, and the
velocity the solver needs is derived from it as ``(x1_hat - xt) / (1 - t)``.
The two are the same model up to a reparameterisation, and it does not make the
``1 / (1 - t)`` factor in the solver go away. What it buys is that the network's
own output is in mel units and anchored to the data manifold at every ``t``,
that the loss is then in those units rather than weighted by ``1 / (1 - t)**2``
towards the end of the trajectory, that an off-trajectory state needs no target
recomputation -- the target is ``x1`` wherever the state is -- and that the
endpoint a discriminator judges is the branch's own output, not the branch's
output diluted by ``(1 - t)`` towards a state that already contains the answer.
The cost turned out to be larger than all of that: a network that emits the
endpoint has no identity path from its state. It never learned to carry the
state's spectral shape through, so the band it produced was flat. The branch can
therefore also emit a velocity instead (``branch_output_param="velocity"``).
The endpoint is then ``y + (1 - t) * v``, and every consumer above still sees
an endpoint.

That last point is why the branch is also trained on states it generated itself.
A field fitted only on the teacher-forced marginal ``xt = x1 * t + noise *
(1 - t)`` is deployed on ``xt + delta``, where ``delta`` is what its own earlier
steps got wrong, and the endpoint-consistent velocity there differs from the one
it was taught by ``delta / (1 - t)`` -- 16x at ``t = 0.938``, the last step of a
16-step solve. The backbone survives this because its conditional mean leans on
the text and the prompt; the branch cannot, because above the cutoff it has
nothing to lean on but the state. So on a fraction of batches the loss is taken
at a state the branch rolled out itself, which is the only term that tells it
how to recover from its own error.

Training is in two stages. Stage 1 is that endpoint MSE plus an optional L1 term
on the assembled endpoint: the L1 says plainly where the endpoint should land
and stabilises early training, at the cost of pulling the branch back towards
the median it was built to avoid, so its weight is a dial and 0 is a supported
setting. Stage 2 is an adversarial finetune, which is what takes the hedge back
out; :meth:`ZipVoiceDual.forward` returns the assembled endpoint and the masks a
discriminator needs so that stage costs no extra backbone pass.

The two models are coupled, not cascaded. Training is joint in the only sense
that matters -- the branch is trained against the frozen backbone it will run
beside, reading that backbone's hidden state at the same timestep through
cross-attention -- while the backbone itself is never updated and never reads
the branch.

Inference solves both at once, from one initial noise and on one timestep grid
(:meth:`ZipVoiceDual.sample`). Two states:

* ``x`` follows the pure backbone trajectory. It never sees the branch, so it is
  bit-identical to what stock ZipVoice would produce for the same noise and
  prompt.
* ``y`` follows the branch, and after every Euler step its observed band is
  overwritten by ``x``'s. That projection is what makes the split
  architectural: the low band of the output, and with it the speaker identity
  the low band carries, is the frozen model's by construction rather than by a
  trained constraint.

The mask is soft -- a raised cosine over a few mel bins, see
:func:`~zipvoice.models.modules.bwe.band_mask` -- so the projection is a blend
rather than a ``where``, and the transition bins are shared between the two
trajectories in proportion to how observed they are.
"""

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from zipvoice.models.modules.band_adapter import (
    FrequencyEncoder,
    RateEncoder,
    SpeakerEncoder,
)
from zipvoice.models.modules.bwe import (
    LOG_MEL_FLOOR,
    band_limit,
    band_mask,
    mel_bin_centers,
    resample_band_limit,
    sample_cutoff,
)
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, make_pad_mask


def endpoint_to_velocity(
    x1_hat: Tensor, xt: Tensor, t: Tensor, eps: float = 1e-3
) -> Tensor:
    """The flow-matching velocity an endpoint prediction implies.

    ``xt + (1 - t) * u == x1_hat`` by construction, so this is the velocity that
    walks straight at the predicted endpoint. On the training marginal it is the
    ordinary conditional target, ``(x1 - xt) / (1 - t) == x1 - noise``; off it,
    it is the velocity that still lands on ``x1_hat`` rather than the one the
    teacher-forced state would have called for, which is the point.

    ``eps`` bounds the ``1 / (1 - t)`` factor. The last step of a 16-step solve
    sits at ``t = 0.9375``, a factor of 16; the clamp only ever bites on a
    timestep far past that.
    """
    return (x1_hat - xt) / (1.0 - t).clamp(min=eps)


class HighBandBranch(nn.Module):
    """A second flow-matching decoder that attends to the first one's state.

    A single Zipformer stack with ``downsampling_factor=(1,)`` -- no U-net. The
    backbone's hidden state is at the feature rate, and keeping the branch there
    too means the cross-attention keys need no resampling and no plumbing
    through ``DownsampledZipformer2Encoder``. It costs some receptive field per
    layer, which matters less here than it would for the backbone: the branch is
    not deciding what is said, only what the said thing sounds like above the
    cutoff, and the cross-attention gives it the backbone's own global view for
    free.

    Its input is the flow-matching state and the same conditioning the adapter
    arm was given, so the comparison between the two arms is about the objective
    rather than about what the model was told.
    """

    def __init__(
        self,
        feat_dim: int,
        memory_dim: int,
        encoder_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        cnn_module_kernel: int = 31,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        time_embed_dim: int = 192,
        freq_dim: int = 64,
        freq_num_layers: int = 2,
        freq_num_heads: int = 4,
        freq_bin_out: int = 4,
        freq_cond_dim: int = 256,
        speaker_cond_dim: int = 128,
        rate_cond_dim: int = 64,
        use_cross_attn: bool = True,
        output_param: str = "endpoint",
    ):
        super().__init__()
        if output_param not in ("endpoint", "velocity"):
            raise ValueError(f"unknown output_param {output_param!r}")
        self.output_param = output_param
        self.feat_dim = feat_dim

        self.freq_encoder = FrequencyEncoder(
            num_bins=feat_dim,
            dim=freq_dim,
            num_layers=freq_num_layers,
            num_heads=freq_num_heads,
            bin_out=freq_bin_out,
            out_dim=freq_cond_dim,
        )
        self.speaker_encoder = SpeakerEncoder(
            num_bins=feat_dim, out_dim=speaker_cond_dim
        )
        self.rate_encoder = RateEncoder(out_dim=rate_cond_dim)

        in_dim = (
            feat_dim * 3 + freq_cond_dim + speaker_cond_dim + rate_cond_dim
        )  # xt, text_condition, speech_condition, then the three streams
        self.net = TTSZipformer(
            in_dim=in_dim,
            out_dim=feat_dim,
            downsampling_factor=(1,),
            num_encoder_layers=num_layers,
            cnn_module_kernel=cnn_module_kernel,
            encoder_dim=encoder_dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
            cross_attn_dim=memory_dim if use_cross_attn else None,
        )
        self.use_cross_attn = use_cross_attn

    def encode_condition(
        self,
        speech_condition: Tensor,
        mask: Tensor,
        cutoff_hz: Tensor,
        prompt_mask: Tensor,
    ) -> Tensor:
        """The three conditioning streams, concatenated, (B, T, C).

        None of the inputs carries the flow-matching state or the timestep, so
        for a given utterance this is one tensor for the whole solve rather than
        one per solver step. It is not cheap -- the frequency encoder runs an
        attention over the mel bins of every frame, B*T sequences of ``num_bins``
        tokens -- so :meth:`~ZipVoiceDual.sample` computes it once and passes it
        back in. Training calls it per batch, where there is one step anyway.
        """
        return torch.cat(
            [
                self.freq_encoder(speech_condition, mask),
                self.speaker_encoder(speech_condition, prompt_mask),
                self.rate_encoder(cutoff_hz, speech_condition.size(1)),
            ],
            dim=2,
        )

    def forward(
        self,
        t: Tensor,
        xt: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        mask: Tensor,
        cutoff_hz: Tensor,
        prompt_mask: Tensor,
        memory: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """The branch's endpoint estimate for its own state.

        Args:
            t: the current timestep, a scalar tensor or of shape (B, 1, 1).
            xt: the branch's flow-matching state, (B, T, F).
            text_condition: the backbone's text condition, (B, T, F).
            speech_condition: the band-limited prompt, zeroed outside it,
                (B, T, F).
            mask: how missing each mel bin is, (B, 1, F), soft.
            cutoff_hz: the prompt's cutoff frequency per utterance, (B,).
            prompt_mask: True at prompt frames, (B, T).
            memory: the backbone's hidden state at this timestep, (B, T, M).
                Required unless the branch was built without cross-attention.
            padding_mask: True at padded frames, (B, T).
            condition: the output of :meth:`encode_condition` for this
                utterance, if it has already been computed. Recomputed here
                when absent.

        Returns:
            The predicted endpoint ``x1_hat``, (B, T, F). Only the bins the band
            mask calls missing are ever used. Pass it through
            :func:`endpoint_to_velocity` for the velocity a solver wants.

            With ``output_param="velocity"`` the network's output is a velocity
            and the endpoint is ``xt + (1 - t) * v``, as the backbone's is. The
            state then reaches the endpoint through an identity path, weighted
            by ``t``. A network that emits the endpoint directly has to rebuild
            the state's spectral detail from scratch. Measured on the endpoint
            arm (every checkpoint from 3k to 28k, sr8000), its endpoint's
            spectral detail was uncorrelated with a 95%-clean state's, against
            0.78 for the backbone. Most of what it produced was one fixed
            per-bin shape. The result is the flat upper band.
        """
        if condition is None:
            condition = self.encode_condition(
                speech_condition, mask, cutoff_hz, prompt_mask
            )
        x = torch.cat([xt, text_condition, speech_condition, condition], dim=2)

        t_state = t
        assert t.dim() in (0, 3), t.shape
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.repeat(x.size(0))

        out = self.net(
            x=x,
            t=t,
            padding_mask=padding_mask,
            memory=memory if self.use_cross_attn else None,
        )
        if self.output_param == "velocity":
            return xt + (1.0 - t_state) * out
        return out


class ZipVoiceDual(ZipVoice):
    """A frozen ZipVoice plus a high-band branch solved alongside it.

    Subclasses :class:`~zipvoice.models.zipvoice.ZipVoice` so that a released
    checkpoint loads into it unchanged -- the backbone parameters keep their
    names -- and so the text front-end, which the branch also needs, is the
    backbone's own rather than a second copy of it.
    """

    def __init__(
        self,
        feat_scale: float = 0.1,
        sampling_rate: int = 24000,
        branch_dim: int = 256,
        branch_num_layers: int = 6,
        branch_num_heads: int = 4,
        branch_feedforward_dim: int = 1024,
        branch_cnn_module_kernel: int = 31,
        freq_cond_dim: int = 256,
        speaker_cond_dim: int = 128,
        rate_cond_dim: int = 64,
        use_cross_attn: bool = True,
        branch_output_param: str = "endpoint",
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
        """
        Args:
            feat_scale: the scale the features are multiplied by before they
                reach the model; the log-mel floor has to be scaled with them.
            sampling_rate: the audio sampling rate the mel filterbank was built
                for, needed to place the mel filter centres in Hz.
            branch_*: the size of :class:`HighBandBranch`.
            use_cross_attn: if False the branch is conditioned exactly as
                before but blind to the backbone's hidden state. This is the
                ablation that says what the fresh keys buy.
            branch_output_param: "endpoint" (the network emits the endpoint)
                or "velocity" (it emits a velocity, and the endpoint is
                ``y + (1 - t) * v``). See :meth:`HighBandBranch.forward`. The
                default keeps checkpoints trained before the option existed
                loading and sampling as they were trained.
            band_*: the band-limiting simulation used during training.
            cutoff_min, cutoff_max, curriculum_*, full_band_prob: the cutoff
                curriculum, see :func:`~zipvoice.models.modules.bwe.sample_cutoff`.
            kwargs: passed to :class:`~zipvoice.models.zipvoice.ZipVoice`.
        """
        super().__init__(**kwargs)

        self.branch = HighBandBranch(
            feat_dim=self.feat_dim,
            memory_dim=self.fm_decoder.encoder_dim,
            encoder_dim=branch_dim,
            num_layers=branch_num_layers,
            num_heads=branch_num_heads,
            feedforward_dim=branch_feedforward_dim,
            cnn_module_kernel=branch_cnn_module_kernel,
            freq_cond_dim=freq_cond_dim,
            speaker_cond_dim=speaker_cond_dim,
            rate_cond_dim=rate_cond_dim,
            use_cross_attn=use_cross_attn,
            output_param=branch_output_param,
        )

        self.register_buffer(
            "mel_centers",
            mel_bin_centers(n_mels=self.feat_dim, sampling_rate=sampling_rate),
            persistent=False,
        )
        self.feat_scale = feat_scale
        self.mel_floor = LOG_MEL_FLOOR * feat_scale
        self.band_transition_bins = band_transition_bins
        self.band_floor_jitter = band_floor_jitter
        self.band_noise_std = band_noise_std
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max
        self.curriculum_beta = curriculum_beta
        self.curriculum_rho = curriculum_rho
        self.full_band_prob = full_band_prob

    def backbone_parameters(self) -> List[nn.Parameter]:
        """Everything the released checkpoint owns, i.e. everything to freeze."""
        return [p for n, p in self.named_parameters() if not n.startswith("branch.")]

    def freeze_backbone(self) -> None:
        for p in self.backbone_parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "ZipVoiceDual":
        """Keep the backbone in eval mode whatever the branch is doing.

        The backbone's layer-drop and dropout schedules are training-time noise
        on a model that is not being trained; leaving them on would make the
        memory the branch attends to, and the trajectory it is being fitted
        against, needlessly different from the ones it meets at inference.
        """
        super().train(mode)
        self.fm_decoder.eval()
        self.text_encoder.eval()
        return self

    def _band_mask(self, cutoff_hz: Tensor) -> Tensor:
        """The deterministic mask, for inference and for the projection."""
        return band_mask(
            cutoff_hz,
            self.mel_centers,
            self.feat_dim,
            transition_bins=self.band_transition_bins,
        )

    def _project(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """Backbone below the cutoff, branch above it, blended across the ramp.

        Written as a displacement from ``x`` rather than as
        ``x * (1 - mask) + y * mask`` so that it is exact and not merely close
        wherever the backbone is meant to win outright: an observed bin returns
        ``x`` bit-for-bit, and a branch that happened to agree with the backbone
        leaves the trajectory untouched. The two forms differ by a rounding, but
        that rounding is the difference between "the low band is the frozen
        model's" as a guarantee and as an approximation.
        """
        return x + (y - x) * mask

    def _memory(
        self,
        t: Tensor,
        y: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """The backbone's hidden state on the branch's own state ``y``.

        The branch reads the backbone's hidden state as its memory, and that
        state has to be computed on ``y``, not on the stock trajectory ``x``.
        At teacher-forced training the two are one state, the flow marginal, so
        the memory carries the band the branch is writing. At inference ``x``
        never receives the branch's writes, so a memory computed on it carries
        nothing above the cutoff: measured on the first dual run (21k, sr8000),
        taking it from ``y`` instead raised 8-12 kHz by 14 dB with SIM-o
        unchanged. The velocity still comes from ``x``, so the low band stays
        the frozen model's.

        Always the conditional pass: the hidden state ``_backbone_step`` hands
        back under guidance is the conditional half's, which is this.
        """
        _, hidden = self.forward_fm_decoder(
            t=t,
            xt=y,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            return_hidden=True,
        )
        return hidden

    def _backbone_step(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        padding_mask: Tensor,
        guidance_scale: Union[float, Tensor] = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """One backbone velocity and the hidden state it came from.

        This is ``DiffusionModel.forward``'s classifier-free guidance, repeated
        here rather than reused because that wrapper chunks the model's single
        return value and cannot carry a second one. The hidden state handed back
        is the conditional half's: the unconditional half is a model that was
        told nothing, and its state is not what the branch should be reading.
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )

        if (guidance_scale == 0.0).all():
            return self.forward_fm_decoder(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                return_hidden=True,
            )

        assert t.dim() == 0, t.shape
        x = torch.cat([x] * 2, dim=0)
        doubled_padding_mask = torch.cat([padding_mask] * 2, dim=0)
        text_condition = torch.cat(
            [torch.zeros_like(text_condition), text_condition], dim=0
        )
        if t > 0.5:
            speech_condition = torch.cat(
                [torch.zeros_like(speech_condition), speech_condition], dim=0
            )
        else:
            guidance_scale = guidance_scale * 2
            speech_condition = torch.cat([speech_condition, speech_condition], dim=0)

        vt, hidden = self.forward_fm_decoder(
            t=t,
            xt=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=doubled_padding_mask,
            return_hidden=True,
        )
        v_uncond, v_cond = vt.chunk(2, dim=0)
        _, h_cond = hidden.chunk(2, dim=0)
        return (1 + guidance_scale) * v_cond - guidance_scale * v_uncond, h_cond

    def forward(
        self,
        tokens: List[List[int]],
        features: Tensor,
        features_lens: Tensor,
        noise: Tensor,
        t: Tensor,
        step: int = 0,
        total_steps: int = 0,
        rollout_prob: float = 0.0,
        rollout_steps: int = 1,
        audio: Optional[Tensor] = None,
        audio_lens: Optional[Tensor] = None,
        fbank: Optional[nn.Module] = None,
        solve_utts: int = 0,
        solve_steps: int = 16,
        solve_t_shift: float = 0.5,
        solve_guidance_scale: float = 1.0,
        fm_space: str = "endpoint",
    ) -> Dict[str, Tensor]:
        """The branch's training terms, over the missing band only.

        With probability ``rollout_prob`` per utterance, both trajectories are
        advanced without gradients from shared pure noise at ``t=0`` to ``t``.
        The supervised branch call then sees the resulting inference-like state
        while retaining the exact clean endpoint target.

        Two terms are returned and the caller weights them:

        ``fm_loss``
            The masked MSE between the predicted and clean endpoints, or with
            ``fm_space="velocity"`` between the velocity the prediction implies
            at the branch's state y, (x1_hat - y)/(1-t), and the one that
            reaches the clean endpoint, (x1 - y)/(1-t). Their difference is the
            endpoint error over (1-t), per utterance, so on teacher rows the
            target is x1 - eps and on rollout rows the detached rollout state
            takes the place of the interpolant. The name is retained for
            checkpoint and metric compatibility.
        ``recon_loss``
            An L1 error on the endpoint estimate. This term is minimised by the
            per-bin conditional median, which in log-mel sits 9-10 dB below the
            energy of any real sample -- it is exactly the hedge the adapter arm
            measured, and weighting it heavily will reproduce that result. It is
            here because it is also the term that stabilises early training by
            saying plainly where the endpoint should land, and because an
            adversarial stage can then take the hedge back out. Weight it
            accordingly, and 0 is a supported setting.

        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, (B, T, F), already scaled.
            features_lens: the length of each feature sequence, (B,).
            noise: the initial noise, (B, T, F).
            t: the timestep, (B, 1, 1).
            step, total_steps: position in training, for the cutoff curriculum.
            rollout_prob: per-utterance probability of replacing the
                teacher-forced state with a detached self-generated state.
            rollout_steps: number of Euler steps from pure noise to ``t``. Must
                be positive when rollout is enabled.
            audio, audio_lens, fbank: when given, the prompt is band-limited by
                really resampling the waveform and re-extracting the mel, which
                is what the evaluation prompts went through, instead of by the
                mel floor. The mask is then the deterministic one inference
                projects with.
            solve_utts: the first this many utterances are also solved from
                pure noise exactly as inference solves them (``solve_steps``,
                ``solve_t_shift``, the backbone under ``solve_guidance_scale``),
                with gradients through the final branch call only. 0 skips it.
            fm_space: "endpoint" or "velocity", where ``fm_loss`` is measured.
                "velocity" needs every ``t`` below 1; the caller bounds it.

        Returns:
            ``fm_loss`` and ``recon_loss``, and alongside them the tensors an
            adversarial stage needs and would otherwise have to recompute:
            ``x1_hat`` (the endpoint estimate, backbone below the cutoff and
            branch above it, as inference assembles it), ``condition`` (the
            band-limited prompt), ``cutoff_hz``, ``band_mask``, ``frame_mask``
            (True where the model had to generate), ``t``, and the realized
            ``rollout_fraction``. With ``solve_utts``, also ``solved_x1``: the
            evaluation-matched output of the first ``solve_utts`` utterances.
        """
        if not 0.0 <= rollout_prob <= 1.0:
            raise ValueError(f"rollout_prob must be in [0, 1], got {rollout_prob}")
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
        if fm_space not in ("endpoint", "velocity"):
            raise ValueError(f"fm_space must be endpoint or velocity, got {fm_space}")

        with torch.no_grad():
            text_condition, padding_mask = self.forward_text_train(
                tokens=tokens,
                features_lens=features_lens,
            )

            cutoff_hz = sample_cutoff(
                batch_size=features.size(0),
                step=step,
                total_steps=total_steps,
                f_min=self.cutoff_min,
                f_max=self.cutoff_max,
                beta=self.curriculum_beta,
                rho=self.curriculum_rho,
                full_band_prob=self.full_band_prob,
                device=features.device,
            )
            if audio is not None:
                band_limited = (
                    resample_band_limit(
                        audio=audio,
                        audio_lens=audio_lens,
                        cutoff_hz=cutoff_hz,
                        fbank=fbank,
                        num_frames=features.size(1),
                    ).to(features.dtype)
                    * self.feat_scale
                )
                mask = self._band_mask(cutoff_hz)
            else:
                band_limited, mask = band_limit(
                    features=features,
                    cutoff_hz=cutoff_hz,
                    centers=self.mel_centers,
                    floor=self.mel_floor,
                    transition_bins=self.band_transition_bins,
                    floor_jitter=self.band_floor_jitter,
                    noise_std=self.band_noise_std,
                )

            # True where the model has to generate, i.e. outside the prompt.
            speech_condition_mask = condition_time_mask(
                features_lens=features_lens,
                mask_percent=(0.7, 1.0),
                max_len=features.size(1),
            )
            speech_condition = torch.where(
                speech_condition_mask.unsqueeze(-1), 0, band_limited
            )

            teacher_xt = features * t + noise * (1 - t)
            prompt_mask = (~speech_condition_mask) & (~padding_mask)

            rollout_mask = torch.rand(
                features.size(0), 1, 1, device=features.device
            ) < rollout_prob
            if rollout_mask.any():
                # Begin exactly where inference does. Ground-truth high-band
                # content is never inserted into a selected rollout input; it
                # appears only as the endpoint supervision below.
                t0 = torch.zeros_like(t)
                dt = t / rollout_steps
                x_roll = noise
                y_roll = x_roll
                rollout_condition = self.branch.encode_condition(
                    speech_condition=speech_condition,
                    mask=mask,
                    cutoff_hz=cutoff_hz,
                    prompt_mask=prompt_mask,
                )
                for rollout_step in range(rollout_steps):
                    tr = dt * rollout_step
                    vr = self.forward_fm_decoder(
                        t=tr,
                        xt=x_roll,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                    )
                    rollout_hidden = self._memory(
                        t=tr,
                        y=y_roll,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                    )
                    rollout_x1 = self.branch(
                        t=tr,
                        xt=y_roll,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        mask=mask,
                        cutoff_hz=cutoff_hz,
                        prompt_mask=prompt_mask,
                        memory=rollout_hidden,
                        padding_mask=padding_mask,
                        condition=rollout_condition,
                    )
                    ur = endpoint_to_velocity(rollout_x1, y_roll, tr)
                    x_roll = x_roll + vr * dt
                    y_roll = self._project(x_roll, y_roll + ur * dt, mask)

                backbone_xt = torch.where(rollout_mask, x_roll, teacher_xt)
                branch_xt = torch.where(rollout_mask, y_roll, teacher_xt)
            else:
                backbone_xt = teacher_xt
                branch_xt = teacher_xt

            # Recompute the frozen backbone at the actual target-time state.
            # Its hidden tensor now has the same state relationship to the
            # branch input that it has inside sample().
            vt, hidden = self.forward_fm_decoder(
                t=t,
                xt=backbone_xt,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                return_hidden=True,
            )
            # The backbone's own endpoint estimate. It is not part of any loss,
            # but it is the low band of the mel an adversarial stage has to
            # judge: showing a discriminator the ground-truth low band instead
            # would hand it a half-real sample to call fake.
            x1_backbone = backbone_xt + (1.0 - t) * vt
            # On teacher rows the two states are one and the hidden state above
            # is already the branch's. On rollout rows it was computed on the
            # stock trajectory, which inference never hands the branch.
            if rollout_mask.any():
                hidden = self._memory(
                    t=t,
                    y=branch_xt,
                    text_condition=text_condition,
                    speech_condition=speech_condition,
                    padding_mask=padding_mask,
                )

        branch_x1 = self.branch(
            t=t,
            xt=branch_xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            mask=mask,
            cutoff_hz=cutoff_hz,
            prompt_mask=prompt_mask,
            memory=hidden,
            padding_mask=padding_mask,
        )

        # Weight each bin by how missing it is: 1.0 above the cutoff, 0.0 below
        # it, and the ramp in between counts in proportion. Frames inside the
        # prompt or the padding count for nothing.
        frame_mask = speech_condition_mask & (~padding_mask)
        weight = mask * frame_mask.to(features.dtype).unsqueeze(-1)
        total = weight.sum().clamp(min=1.0)
        err = branch_x1.float() - features.float()
        if fm_space == "velocity":
            # Per utterance, before the reduction; float32 because at t near
            # t_max the factor is large.
            err = err / (1.0 - t.float())
        fm_loss = (weight * err**2).sum() / total

        # Assemble the endpoint exactly as inference does: backbone below the
        # cutoff and the branch's direct endpoint prediction above it.
        x1_hat = self._project(x1_backbone, branch_x1, mask)
        recon_loss = (weight * (x1_hat - features).abs()).sum() / total

        extra = {}
        if solve_utts > 0:
            k = min(solve_utts, features.size(0))
            extra["solved_x1"] = self._solve(
                noise=noise[:k],
                text_condition=text_condition[:k],
                speech_condition=speech_condition[:k],
                padding_mask=padding_mask[:k],
                mask=mask[:k],
                cutoff_hz=cutoff_hz[:k],
                prompt_mask=prompt_mask[:k],
                num_step=solve_steps,
                t_shift=solve_t_shift,
                guidance_scale=solve_guidance_scale,
                grad_last_step=True,
            )

        return {
            **extra,
            "fm_loss": fm_loss,
            "recon_loss": recon_loss,
            "x1_hat": x1_hat,
            "condition": speech_condition,
            "cutoff_hz": cutoff_hz,
            "band_mask": mask,
            "frame_mask": frame_mask,
            "t": t,
            "rollout_fraction": rollout_mask.float().mean(),
        }

    def _solve(
        self,
        noise: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        padding_mask: Tensor,
        mask: Tensor,
        cutoff_hz: Tensor,
        prompt_mask: Tensor,
        num_step: int,
        t_shift: float,
        guidance_scale: float,
        grad_last_step: bool = False,
    ) -> Tensor:
        """Solve both trajectories from ``noise`` and return the assembled mel.

        This is the whole of inference once the conditioning is built, and
        training calls it too so that an auxiliary loss can see exactly the mel
        evaluation scores. With ``grad_last_step`` every step but the last runs
        without gradients and the last branch call keeps them. The last Euler
        step lands on t=1, where it equals that call's endpoint estimate, so the
        high band of the returned mel *is* that call's output: the gradient is
        exact at the output and truncated through the states that led to it.
        """
        timesteps = get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=num_step,
            t_shift=t_shift,
            device=noise.device,
        )
        x = noise
        y = x  # one noise, so the two trajectories start with the same low band

        for step in range(num_step):
            t = timesteps[step]
            dt = timesteps[step + 1] - timesteps[step]
            with_grad = grad_last_step and step == num_step - 1
            no_grad = grad_last_step and not with_grad

            with torch.set_grad_enabled(torch.is_grad_enabled() and not no_grad):
                if step == 0 or with_grad:
                    # Conditioning depends on the prompt and the cutoff, neither
                    # of which moves along the trajectory, so it is computed
                    # once rather than per step -- and once more for the step
                    # that has to carry gradients into the encoder.
                    condition = self.branch.encode_condition(
                        speech_condition=speech_condition,
                        mask=mask,
                        cutoff_hz=cutoff_hz,
                        prompt_mask=prompt_mask,
                    )
                with torch.no_grad() if grad_last_step else nullcontext():
                    # The frozen backbone never needs gradients, but plain
                    # inference leaves grad mode to its caller.
                    v, _ = self._backbone_step(
                        t=t,
                        x=x,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                        guidance_scale=guidance_scale,
                    )
                    hidden = self._memory(
                        t=t,
                        y=y,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                    )
                branch_x1 = self.branch(
                    t=t,
                    xt=y,
                    text_condition=text_condition,
                    speech_condition=speech_condition,
                    mask=mask,
                    cutoff_hz=cutoff_hz,
                    prompt_mask=prompt_mask,
                    memory=hidden,
                    padding_mask=padding_mask,
                    condition=condition,
                )
                u = endpoint_to_velocity(branch_x1, y, t)

                x = x + v * dt  # the stock trajectory, which never reads the branch
                y = self._project(x, y + u * dt, mask)

        return self._project(x, y, mask)

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: Tensor,
        prompt_features_lens: Tensor,
        cutoff_hz: Union[float, Tensor],
        features_lens: Optional[Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Solve both models from one noise and one timestep grid.

        Args:
            cutoff_hz: the prompt's own cutoff frequency, a float in Hz or a
                tensor of shape (B,). This is a property of the audio that was
                handed in, not a dial: it says which bins the prompt actually
                contains and therefore where the two trajectories are joined.
            everything else: as in :meth:`ZipVoice.sample`.

        Returns:
            As in :meth:`ZipVoice.sample`: the generated features without the
            prompt, their lengths, the prompt part, and its lengths.
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

        speech_condition = torch.nn.functional.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B, T, F)
        # False means speech condition positions, i.e. the prompt.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        if not torch.is_tensor(cutoff_hz):
            cutoff_hz = torch.full(
                (batch_size,),
                float(cutoff_hz),
                device=device,
                dtype=prompt_features.dtype,
            )
        cutoff_hz = cutoff_hz.to(device=device, dtype=prompt_features.dtype)
        mask = self._band_mask(cutoff_hz)

        noise = torch.randn(
            batch_size, num_frames, prompt_features.size(-1), device=device
        )
        prompt_mask = (~speech_condition_mask) & (~padding_mask)

        x1 = self._solve(
            noise=noise,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            mask=mask,
            cutoff_hz=cutoff_hz,
            prompt_mask=prompt_mask,
            num_step=num_step,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
        )

        x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
        x1_prompt = torch.zeros(
            x1.size(0), prompt_features_lens.max(), x1.size(2), device=x1.device
        )
        x1_wo_prompt = torch.zeros(
            x1.size(0), x1_wo_prompt_lens.max(), x1.size(2), device=x1.device
        )
        for i in range(x1.size(0)):
            x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                i,
                prompt_features_lens[i] : prompt_features_lens[i]
                + x1_wo_prompt_lens[i],
            ]
            x1_prompt[i, : prompt_features_lens[i], :] = x1[
                i, : prompt_features_lens[i]
            ]

        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens
