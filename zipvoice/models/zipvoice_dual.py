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

Training is in two stages. Stage 1 is that flow-matching MSE plus an optional
L1 term on the endpoint estimate: the L1 says plainly where the endpoint should
land and stabilises early training, at the cost of pulling the branch back
towards the median it was built to avoid, so its weight is a dial and 0 is a
supported setting. Stage 2 is an adversarial finetune, which is what takes the
hedge back out; :meth:`ZipVoiceDual.forward` returns the assembled endpoint and
the masks a discriminator needs so that stage costs no extra backbone pass.

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
    sample_cutoff,
)
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, make_pad_mask


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
    ):
        super().__init__()
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
        """Velocity for the branch's own state.

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
            The predicted velocity, (B, T, F). Only the bins the band mask calls
            missing are ever used.
        """
        if condition is None:
            condition = self.encode_condition(
                speech_condition, mask, cutoff_hz, prompt_mask
            )
        x = torch.cat([xt, text_condition, speech_condition, condition], dim=2)

        assert t.dim() in (0, 3), t.shape
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.repeat(x.size(0))

        return self.net(
            x=x,
            t=t,
            padding_mask=padding_mask,
            memory=memory if self.use_cross_attn else None,
        )


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
        )

        self.register_buffer(
            "mel_centers",
            mel_bin_centers(n_mels=self.feat_dim, sampling_rate=sampling_rate),
            persistent=False,
        )
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
    ) -> Dict[str, Tensor]:
        """The branch's training terms, over the missing band only.

        The backbone is teacher-forced exactly as it is in stock training, so
        the two models share ``xt`` and the projection inference performs is a
        no-op here; there is nothing extra to run on the training side.

        Two terms are returned and the caller weights them:

        ``fm_loss``
            The flow-matching MSE on velocity. On its own it asks the branch to
            be a generative model of the missing band, which is the whole point
            of the architecture: a sample from a velocity field does not
            average, so it does not hedge.
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

        Returns:
            ``fm_loss`` and ``recon_loss``, and alongside them the tensors an
            adversarial stage needs and would otherwise have to recompute:
            ``x1_hat`` (the endpoint estimate, backbone below the cutoff and
            branch above it, as inference assembles it), ``condition`` (the
            band-limited prompt), ``cutoff_hz``, ``band_mask``, ``frame_mask``
            (True where the model had to generate) and ``t``.
        """
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

            xt = features * t + noise * (1 - t)

            vt, hidden = self.forward_fm_decoder(
                t=t,
                xt=xt,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                return_hidden=True,
            )
            # The backbone's own endpoint estimate. It is not part of any loss,
            # but it is the low band of the mel an adversarial stage has to
            # judge: showing a discriminator the ground-truth low band instead
            # would hand it a half-real sample to call fake.
            x1_backbone = xt + (1.0 - t) * vt

        ut = features - noise
        u_hat = self.branch(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            mask=mask,
            cutoff_hz=cutoff_hz,
            prompt_mask=(~speech_condition_mask) & (~padding_mask),
            memory=hidden,
            padding_mask=padding_mask,
        )

        # Weight each bin by how missing it is: 1.0 above the cutoff, 0.0 below
        # it, and the ramp in between counts in proportion. Frames inside the
        # prompt or the padding count for nothing.
        frame_mask = speech_condition_mask & (~padding_mask)
        weight = mask * frame_mask.to(features.dtype).unsqueeze(-1)
        total = weight.sum().clamp(min=1.0)
        fm_loss = (weight * (u_hat - ut) ** 2).sum() / total

        # The endpoint the branch's velocity points at, assembled the way
        # inference assembles it: backbone below the cutoff, branch above it.
        x1_hat = self._project(x1_backbone, xt + (1.0 - t) * u_hat, mask)
        recon_loss = (weight * (x1_hat - features).abs()).sum() / total

        return {
            "fm_loss": fm_loss,
            "recon_loss": recon_loss,
            "x1_hat": x1_hat,
            "condition": speech_condition,
            "cutoff_hz": cutoff_hz,
            "band_mask": mask,
            "frame_mask": frame_mask,
            "t": t,
        }

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

        x = torch.randn(
            batch_size, num_frames, prompt_features.size(-1), device=device
        )
        y = x  # one noise, so the two trajectories start with the same low band

        timesteps = get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=num_step,
            t_shift=t_shift,
            device=device,
        )
        prompt_mask = (~speech_condition_mask) & (~padding_mask)

        # Conditioning depends on the prompt and the cutoff, neither of which
        # moves along the trajectory, so it is computed once for all num_step
        # steps rather than rebuilt inside the loop.
        condition = self.branch.encode_condition(
            speech_condition=speech_condition,
            mask=mask,
            cutoff_hz=cutoff_hz,
            prompt_mask=prompt_mask,
        )

        for step in range(num_step):
            t = timesteps[step]
            dt = timesteps[step + 1] - timesteps[step]

            v, hidden = self._backbone_step(
                t=t,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
            )
            u = self.branch(
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

            x = x + v * dt  # the stock trajectory, which never reads the branch
            y = self._project(x, y + u * dt, mask)

        x1 = self._project(x, y, mask)

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
