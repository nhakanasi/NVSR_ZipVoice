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

"""Distribution-matching distillation of a one-pass band-limited student.

The student is the deployed coupled adapter: frozen released ZipVoice, the band
branch, and the gated decoder LoRA, sampled in a single trajectory with one
backbone forward per step -- no projection onto the frozen model's low band and
no decoupling. Every earlier way of training that generator scored a one-step
endpoint estimate from a teacher-forced state (blind to the deficit the 16-step
solve produces), or regressed the solve onto a target it is not paired with
(there is no ground-truth high band for a *generated* low band). This module
supervises the solved sample itself, at the level of distributions, after DMD /
DMD2 (Yin et al., CVPR 2024 / NeurIPS 2024):

    grad_x  ~  s_fake(x_t) - s_real(x_t),     x = student sample, x_t = noised x

* ``s_real`` is the released ZipVoice conditioned on the **clean, full-band**
  prompt, with classifier-free guidance. Its distribution is what a clean
  recording would have produced: the right speaker over every bin. It costs no
  extra weights: the adapter model with a zero band mask *is* the released
  checkpoint (LoRA gated off, branch residual zero), and here it is run through
  the base-class decoder call so the branch is not even evaluated.
* ``s_fake`` tracks the student's own output distribution under the same
  condition. It is the same frozen backbone plus a second LoRA, owned by the
  trainer (:class:`FakeScoreLoRA`) and trained with plain flow matching on the
  student's detached samples. It starts at zero, so ``s_fake == s_real`` at
  step 0 and the generator signal grows only as the fake score learns where the
  student actually is.

Because the comparison is between distributions, nothing has to be paired with
the generated low band; and because ``x`` is the solved sample, the loss sees
exactly the trajectory deployment runs. Identity and band level are one target
rather than two losses to balance, and there is no speaker-verifier gradient to
exploit.

Kept out of the model on purpose, like the discriminators: the checkpoint stays
the released tensors plus the branch plus the student LoRA, so inference and
model averaging load it as an ordinary ``ZipVoiceAdapter``.
"""

import contextlib
import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipvoice.models.modules.bwe import resample_band_limit
from zipvoice.models.modules.solver import DiffusionModel, get_time_steps
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_adapter import LORA_TARGETS


class FakeScoreLoRA(nn.Module):
    """A second low-rank update on the backbone's decoder, for the fake score.

    The factors live here rather than on the Linears they modify, and the
    forward hooks they ride on are inert unless :meth:`enabled` is active. So
    the model's own state dict never sees them and every other caller of the
    decoder -- the student, the teacher, inference -- is unaffected.

    ``lora_B`` starts at exactly zero (this module is optimised with AdamW, not
    ScaledAdam, so a zero tensor can move), which makes the fake score equal to
    the teacher at initialisation.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        alpha: Optional[float] = None,
        targets: Sequence[str] = LORA_TARGETS,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.active = False
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self._handles = []
        for name, module in model.fm_decoder.named_modules():
            if not (
                isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
            ):
                continue
            key = name.replace(".", "__")
            weight = module.weight
            self.lora_A[key] = nn.Parameter(
                torch.randn(rank, module.in_features, dtype=weight.dtype)
                / math.sqrt(module.in_features)
            )
            self.lora_B[key] = nn.Parameter(
                torch.zeros(module.out_features, rank, dtype=weight.dtype)
            )
            self._handles.append(module.register_forward_hook(partial(self._hook, key)))
        if not self._handles:
            raise ValueError("no decoder Linear matched the LoRA targets")

    def _hook(self, key: str, module: nn.Linear, inputs, output):
        if not self.active:
            return output
        update = F.linear(F.linear(inputs[0], self.lora_A[key]), self.lora_B[key])
        return output + update * self.scale

    @contextlib.contextmanager
    def enabled(self):
        previous = self.active
        self.active = True
        try:
            yield
        finally:
            self.active = previous

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _Backbone:
    """The released decoder alone, as a ``DiffusionModel`` target.

    Calls ``ZipVoice.forward_fm_decoder`` -- the base-class method -- on the
    adapter model, so the branch is never evaluated and the student LoRA's gate
    stays None (off). Whatever :class:`FakeScoreLoRA` is enabled still applies,
    since it hooks the same Linears.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def velocity(self, t, xt, text_condition, speech_condition, padding_mask=None):
        return ZipVoice.forward_fm_decoder(
            self.model,
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )


def backbone_velocity(
    model: nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    text_condition: torch.Tensor,
    speech_condition: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """Velocity of the released decoder, with the solver's guidance rule.

    Guidance goes through ``DiffusionModel`` itself, so the unconditional half
    and the below-0.5 rescaling are exactly what the deployed sampler does.
    ``t`` must be a 0-dim tensor when ``guidance_scale`` is non-zero.
    """
    wrapper = DiffusionModel(_Backbone(model), func_name="velocity")
    return wrapper(
        t=t,
        x=x,
        text_condition=text_condition,
        speech_condition=speech_condition,
        padding_mask=padding_mask,
        guidance_scale=guidance_scale,
    )


def student_rollout(
    model: nn.Module,
    noise: torch.Tensor,
    conditioning: Dict[str, torch.Tensor],
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
    grad_steps: int = 1,
) -> torch.Tensor:
    """The deployed one-pass solve, graphed on its last ``grad_steps`` steps.

    Same call path as ``ZipVoiceAdapter.sample`` without projection or
    decoupling: ``model.solver.model`` (so guidance is the deployed rule), the
    coupled residual, and the previous step's residual carried into the
    decoder's LoRA input. Unlike ``identity_rollout``, the backbone is *not*
    detached on the graphed steps: the student LoRA lives in the decoder, and
    it is what has to learn to read the state the branch writes.

    Dropout is off along the whole path, as at inference; modes are restored.
    """
    if not 1 <= grad_steps <= num_step:
        raise ValueError(f"grad_steps {grad_steps} outside [1, {num_step}]")
    if model.decouple_residual or model.project_low_band or model.band_fill:
        raise ValueError(
            "The DMD student is the coupled one-pass solve; decoupling, "
            "projection and band_fill are different generators"
        )
    grid = get_time_steps(
        t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift, device=noise.device
    )
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    model._carry_residual = True
    model._prev_residual = None
    model._decouple_delta = None
    model._decouple_prev = None
    model._bf_state = None
    x = noise
    first_graphed = num_step - grad_steps
    try:
        for step in range(num_step):
            with torch.set_grad_enabled(step >= first_graphed and torch.is_grad_enabled()):
                velocity = model.solver.model(
                    t=grid[step],
                    x=x,
                    guidance_scale=guidance_scale,
                    **conditioning,
                )
                x = x + velocity * (grid[step + 1] - grid[step])
    finally:
        model._carry_residual = False
        model._prev_residual = None
        for module, mode in modes:
            module.training = mode
    return x


@contextlib.contextmanager
def _eval_mode(model: nn.Module):
    """Dropout and stochastic depth off, restored afterwards.

    The two scores are differenced, so they must be the same deterministic
    function of the backbone up to the fake LoRA: with dropout on, each forward
    draws its own masks and the difference measures those instead.
    """
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``value`` (B, T, F) over the frames ``mask`` (B, T) selects."""
    weight = mask.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum() / (weight.sum() * value.size(-1)).clamp(min=1.0)


def dmd_generator_loss(
    model: nn.Module,
    fake: FakeScoreLoRA,
    sample: torch.Tensor,
    batch: "DMDBatch",
    t_min: float = 0.02,
    t_max: float = 0.98,
    real_guidance: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DMD2's distribution-matching loss on the student's solved sample.

    One noise level for the whole call, because guidance needs a scalar ``t``.
    The prompt frames of the sample are replaced by the clean prompt before
    noising: they are discarded at inference, and the teacher was trained with
    the real prompt there.

    The gradient that reaches ``sample`` is ``(x1_fake - x1_real) / norm`` over
    the generated frames, ``norm`` being the per-utterance mean absolute gap
    between the sample and the teacher's endpoint (DMD's normaliser), so the
    step is scale-free across noise levels and utterances.
    """
    device = sample.device
    t = torch.empty((), device=device).uniform_(t_min, t_max)
    prompt = batch.prompt_mask.unsqueeze(-1)
    gen = batch.gen_mask
    with torch.no_grad(), _eval_mode(model):
        x1 = torch.where(prompt, batch.features, sample.detach())
        noise = torch.randn_like(x1)
        xt = x1 * t + noise * (1.0 - t)
        common = dict(
            x=xt,
            text_condition=batch.text_condition,
            speech_condition=batch.clean_condition,
            padding_mask=batch.padding_mask,
        )
        v_real = backbone_velocity(model, t=t, guidance_scale=real_guidance, **common)
        with fake.enabled():
            v_fake = backbone_velocity(model, t=t, guidance_scale=0.0, **common)
        x1_real = xt + (1.0 - t) * v_real
        x1_fake = xt + (1.0 - t) * v_fake

        weight = gen.unsqueeze(-1).to(x1.dtype)
        per_utt = (weight.sum(dim=(1, 2)) * x1.size(-1)).clamp(min=1.0)
        norm = ((x1 - x1_real).abs() * weight).sum(dim=(1, 2)) / per_utt
        grad = (x1_fake - x1_real) / norm.clamp(min=1e-4).view(-1, 1, 1)
        grad = torch.nan_to_num(grad) * weight
        target = sample.detach() - grad

    loss = 0.5 * _masked_mean((sample - target).pow(2), gen)
    stats = {
        "t": float(t),
        "grad_rms": float(_masked_mean(grad.pow(2), gen).sqrt()),
        "norm": float(norm.mean()),
        "fake_real_gap": float(_masked_mean((x1_fake - x1_real).pow(2), gen).sqrt()),
    }
    return loss, stats


def fake_score_loss(
    model: nn.Module,
    fake: FakeScoreLoRA,
    sample: torch.Tensor,
    batch: "DMDBatch",
) -> torch.Tensor:
    """Flow matching for the fake score on the student's detached samples.

    Conditioned on the clean prompt, like the teacher, so that the two scores
    differ only in which distribution of outputs they were fitted to. No
    guidance: the fake score is only ever queried unguided.
    """
    sample = sample.detach()
    x1 = torch.where(batch.prompt_mask.unsqueeze(-1), batch.features, sample)
    t = torch.rand(x1.size(0), 1, 1, device=x1.device)
    noise = torch.randn_like(x1)
    xt = x1 * t + noise * (1.0 - t)
    with fake.enabled(), _eval_mode(model):
        v = ZipVoice.forward_fm_decoder(
            model,
            t=t,
            xt=xt,
            text_condition=batch.text_condition,
            speech_condition=batch.clean_condition,
            padding_mask=batch.padding_mask,
        )
    return _masked_mean((v - (x1 - noise)).pow(2), batch.gen_mask)


class DMDBatch:
    """Everything one DMD step needs, cropped to the utterances it runs on."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def student_conditioning(self) -> Dict[str, torch.Tensor]:
        return dict(
            text_condition=self.text_condition,
            speech_condition=self.student_condition,
            padding_mask=self.padding_mask,
            band_mask=self.band_mask,
            cutoff_hz=self.cutoff_hz,
            prompt_mask=self.prompt_mask,
        )


def build_dmd_batch(
    model: nn.Module,
    tokens: List[List[int]],
    features: torch.Tensor,
    features_lens: torch.Tensor,
    cutoffs: Sequence[float],
    audio: Optional[torch.Tensor] = None,
    audio_lens: Optional[torch.Tensor] = None,
    fbank: Optional[nn.Module] = None,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    num_utts: int = 1,
    max_frames: int = 0,
    prompt_frac: Tuple[float, float] = (0.2, 0.5),
) -> DMDBatch:
    """Pick utterances, band-limit their prompts and build both conditions.

    The prompt is a prefix, as at inference, covering ``prompt_frac`` of each
    utterance; the rest is the span to generate. Text is the whole utterance's,
    at its true duration, which is what inference gives the model too (prompt
    text plus target text over the prompt plus predicted length).

    The degraded prompt is made by really resampling the waveform when
    ``audio`` and ``fbank`` are given, which is what evaluation prompts are;
    otherwise by the mel-domain floor.
    """
    device = features.device
    lens = features_lens
    eligible = torch.ones_like(lens, dtype=torch.bool)
    if max_frames > 0:
        eligible = lens <= max_frames
        if not bool(eligible.any()):
            eligible = lens == lens.min()
    choices = eligible.nonzero().flatten()
    choices = choices[torch.randperm(len(choices), device=choices.device)[:num_utts]]
    index = choices.tolist()
    lens = lens[choices]
    frames = int(lens.max())
    feats = features[choices, :frames]
    toks = [tokens[i] for i in index]
    cutoff_hz = torch.tensor(
        [float(cutoffs[j]) for j in torch.randint(len(cutoffs), (len(index),))],
        device=device,
    )

    _, band_mask = model._band_limit(feats, cutoff_hz, randomize=False)
    if audio is not None and fbank is not None:
        with torch.no_grad():
            degraded = resample_band_limit(
                audio=audio[choices].to(device),
                audio_lens=audio_lens[choices].to(device),
                cutoff_hz=cutoff_hz,
                fbank=fbank,
                num_frames=features.size(1),
                sampling_rate=sampling_rate,
            )
        degraded = degraded[:, :frames].to(feats.dtype) * feat_scale
    else:
        degraded, _ = model._band_limit(feats, cutoff_hz, randomize=False)

    with torch.no_grad():
        text_condition, padding_mask = model.forward_text_train(
            tokens=toks, features_lens=lens
        )
    lo, hi = prompt_frac
    frac = torch.empty(len(index), device=device).uniform_(lo, hi)
    prompt_lens = (frac * lens.to(frac.dtype)).round().long().clamp(min=1)
    prompt_lens = torch.minimum(prompt_lens, lens - 1).clamp(min=1)
    positions = torch.arange(frames, device=device).unsqueeze(0)
    prompt_mask = (positions < prompt_lens.unsqueeze(1)) & ~padding_mask
    gen_mask = ~prompt_mask & ~padding_mask
    zero = torch.zeros_like(feats)
    return DMDBatch(
        index=index,
        features=feats,
        features_lens=lens,
        text_condition=text_condition,
        padding_mask=padding_mask,
        prompt_mask=prompt_mask,
        gen_mask=gen_mask,
        band_mask=band_mask,
        cutoff_hz=cutoff_hz,
        clean_condition=torch.where(prompt_mask.unsqueeze(-1), feats, zero),
        student_condition=torch.where(prompt_mask.unsqueeze(-1), degraded, zero),
    )


class DMDController:
    """The trainer-side half of DMD: the fake score, its optimiser and schedule.

    One controller per run. Each training batch calls :meth:`generator_loss`
    (returns None on batches that are not generator batches) before the
    student's backward, and :meth:`fake_update` after the student's optimiser
    step. The fake score therefore always trains on the student as it now is,
    and on generator batches it reuses that batch's sample instead of solving
    again. Two time scales, as in DMD2: the fake score is updated on every
    batch, the generator every ``gen_interval`` batches, and not at all before
    ``start_step`` -- before then the fake score is still the teacher, and the
    only signal would be the guidance direction.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        lr: float = 1e-4,
        loss_weight: float = 1.0,
        start_step: int = 1000,
        gen_interval: int = 5,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        real_guidance: float = 1.0,
        t_shift: float = 0.5,
        grad_steps: int = 1,
        t_min: float = 0.02,
        t_max: float = 0.98,
        num_utts: int = 1,
        max_frames: int = 0,
        cutoffs: Sequence[float] = (4000.0, 8000.0),
        prompt_frac: Tuple[float, float] = (0.2, 0.5),
        feat_scale: float = 0.1,
        sampling_rate: int = 24000,
        use_fp16: bool = False,
    ):
        device = next(model.parameters()).device
        self.fake = FakeScoreLoRA(model, rank=rank).to(device)
        self.optimizer = torch.optim.AdamW(
            self.fake.parameters(), lr=lr, betas=(0.0, 0.999), weight_decay=0.0
        )
        self.scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
        self.loss_weight = loss_weight
        self.start_step = start_step
        self.gen_interval = max(1, gen_interval)
        self.rollout = dict(
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            grad_steps=grad_steps,
        )
        self.real_guidance = real_guidance
        self.t_range = (t_min, t_max)
        self.batch_args = dict(
            cutoffs=tuple(cutoffs),
            num_utts=num_utts,
            max_frames=max_frames,
            prompt_frac=prompt_frac,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
        )
        self._pending = None

    def is_generator_step(self, step: int) -> bool:
        return step >= self.start_step and (step - self.start_step) % self.gen_interval == 0

    def _batch(self, model, tokens, features, features_lens, audio, audio_lens, fbank):
        return build_dmd_batch(
            model,
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            audio=audio,
            audio_lens=audio_lens,
            fbank=fbank,
            **self.batch_args,
        )

    def generator_loss(
        self,
        model: nn.Module,
        step: int,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        fbank: Optional[nn.Module] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """The weighted generator loss for this batch, or None off-schedule.

        Builds a DMD batch, solves it with a graph on the last steps, and
        leaves the detached sample for :meth:`fake_update`.
        """
        self._pending = None
        if not self.is_generator_step(step):
            return None, {}
        batch = self._batch(model, tokens, features, features_lens, audio, audio_lens, fbank)
        sample = student_rollout(
            model,
            torch.randn_like(batch.features),
            batch.student_conditioning(),
            **self.rollout,
        )
        loss, stats = dmd_generator_loss(
            model,
            self.fake,
            sample,
            batch,
            t_min=self.t_range[0],
            t_max=self.t_range[1],
            real_guidance=self.real_guidance,
        )
        self._pending = (batch, sample.detach())
        stats["loss"] = float(loss.detach())
        stats["frames"] = float(batch.gen_mask.sum())
        return self.loss_weight * loss, stats

    def fake_update(
        self,
        model: nn.Module,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        fbank: Optional[nn.Module] = None,
        autocast=contextlib.nullcontext,
    ) -> Dict[str, float]:
        """One flow-matching step of the fake score on the current student."""
        if self._pending is not None:
            batch, sample = self._pending
        else:
            batch = self._batch(
                model, tokens, features, features_lens, audio, audio_lens, fbank
            )
            with torch.no_grad(), autocast():
                sample = student_rollout(
                    model,
                    torch.randn_like(batch.features),
                    batch.student_conditioning(),
                    **dict(self.rollout, grad_steps=1),
                )
        self._pending = None
        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss = fake_score_loss(model, self.fake, sample, batch)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        # The fake score's loss runs the frozen backbone; nothing of the
        # student should have picked up a gradient from it.
        return {"fake_loss": float(loss.detach()), "fake_frames": float(batch.gen_mask.sum())}

    def state_dict(self) -> Dict:
        return {
            "fake": self.fake.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.fake.load_state_dict(state["fake"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
