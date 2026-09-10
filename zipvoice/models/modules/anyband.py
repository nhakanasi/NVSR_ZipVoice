# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
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

"""Frequency-aware conditioning for ZipVoice, after AnyBand.

This module implements the conditioning half of "AnyBand: Unified
Multi-Bandwidth Speech Extension via Frequency-Aware In-Context Spectral
Infilling" (arXiv 2608.00572), adapted to a text-to-speech flow-matching model
rather than a bandwidth-extension one.

AnyBand's observation is that a frame-wise projection of a mel spectrogram
destroys the structure that matters for deciding what belongs in a missing
band: the relationship *between* frequency bins. Its remedy is to treat the mel
bins of a single frame as a token sequence and run self-attention along the
frequency axis, independently per frame, so that an observed low band can speak
directly to the bins that have to be invented above it. The frequency encoder
here is that construction; the frequency decoder is its mirror on the read-out
side.

Three things differ from the paper, each forced by the setting:

1. AnyBand extracts F0 and voicing from the band-limited *input waveform* with
   pYIN. In text-to-speech there is no waveform over the span being generated,
   so :class:`MelF0Estimator` predicts the same two quantities from a mel
   instead. It is trained separately against pYIN targets and frozen thereafter.

2. AnyBand's frequency encoder runs at every ODE step because the noisy state is
   one of its inputs. That is kept, but the F0 and voicing contour is extracted
   once, at the first phase boundary, from the model's own endpoint estimate.

3. The blend between the base conditioning and the frequency conditioning is not
   a fixed schedule but a single learned, monotone function of the flow-matching
   timestep (:class:`TimestepGate`), so the model decides for itself how much of
   each to use at each point along the trajectory.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# F0 normalisation range, in Hz, following AnyBand.
F0_MIN_HZ = 86.0
F0_MAX_HZ = 1100.0


def normalize_f0(f0_hz: Tensor) -> Tensor:
    """Map an F0 contour in Hz onto [0, 1] on a log scale.

    Frames with a non-positive F0 are unvoiced and map to exactly 0; the binary
    voicing flag, not this value, is what tells the model they are unvoiced.
    """
    voiced = f0_hz > 0
    log_f0 = torch.log(f0_hz.clamp(min=1e-5))
    lo, hi = math.log(F0_MIN_HZ), math.log(F0_MAX_HZ)
    normalized = ((log_f0 - lo) / (hi - lo)).clamp(0.0, 1.0)
    return torch.where(voiced, normalized, torch.zeros_like(normalized))


class FreqSelfAttention(nn.Module):
    """Multi-head self-attention along the frequency axis of one frame.

    The input is folded so that every (utterance, frame) pair becomes an
    independent sequence of ``num_bins`` tokens. That fold makes the batch very
    large -- a 1000-frame utterance at batch size 8 gives 8000 sequences -- so
    the attention matrix is never materialised: it goes through
    ``scaled_dot_product_attention``, which keeps memory linear in the sequence
    length.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, num_bins, dim) -> (N, num_bins, dim)."""
        n, s, _ = x.shape
        qkv = self.qkv(x).reshape(n, s, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(n, s, -1)
        return self.proj(out)


class FreqTransformerBlock(nn.Module):
    """Pre-norm Transformer block whose attention runs over frequency."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FreqSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class FrequencyEncoder(nn.Module):
    """AnyBand's frequency encoder.

    Every mel bin of every frame is a token carrying three values: the bin's
    value in the noisy flow-matching state, its value in the clean band-limited
    condition, and how far the condition is observed there. Without that third
    value a missing bin and a genuinely silent one are indistinguishable, since
    both sit at the log-mel floor. Attention along the
    frequency axis then lets the observed bins inform the missing ones within
    that frame.

    The frequency axis is then folded into the channel axis rather than pooled
    away. An earlier version ended in learned-query attention pooling, which
    collapsed all ``num_bins`` tokens to a single vector before the temporal
    backbone saw them: every bin's identity was destroyed at exactly the point
    the branch was supposed to be delivering per-bin detail, leaving only a
    per-frame summary. Here each token is instead read out to ``bin_out``
    values, the result is flattened over frequency, and one linear layer maps
    ``num_bins * bin_out`` to the backbone width. Bin identity survives as
    position in the flattened vector, which is the same way the backbone's own
    ``in_proj`` consumes a mel. Because the blocks above have already given
    every token global context along frequency, no separate pooled summary is
    needed.

    ``out_proj`` is the single point at which this branch enters the pretrained
    backbone, so it is also the branch's scale knob; it is deliberately not
    zero-initialised. See ``ZipVoiceAnyBand.__init__`` on why.

    The condition is passed in already band-limited and is never noised,
    matching the paper's treatment of the observed region.
    """

    def __init__(
        self,
        num_bins: int,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        bin_out: int = 4,
        out_dim: int = 512,
        dropout: float = 0.0,
        ltas_channels: int = 4,
        use_state: bool = True,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.ltas_channels = ltas_channels
        # Whether the noisy flow-matching state is one of the per-bin inputs.
        # It is when this encoder conditions the backbone's own vector field,
        # since the correction then has to depend on where the trajectory
        # currently is. Used as a side branch's feature extractor over the
        # condition alone -- ``ZipVoiceAdapter`` -- it is not: there the state
        # enters through the backbone's hidden output, which the adapter
        # cross-attends to, and reading it here as well would only duplicate it.
        self.use_state = use_state
        # The long-term average spectrum of the prompt, convolved along
        # frequency. Every other input this encoder reads is per frame, and the
        # attention runs over bins inside a single frame, so the branch has no
        # way to form a quantity pooled over time however many layers it is
        # given. The recording chain -- microphone, codec, room -- is exactly
        # such a quantity, and it is what decides the shape of the band that has
        # to be invented.
        #
        # A convolution rather than a per-bin projection because its receptive
        # field crosses the cutoff: it can read the slope of the observed
        # envelope and continue it into the bins that were removed, which is the
        # global half of bandwidth extension. The second input channel is the
        # observed mask, without which the zeros standing for unmeasured bins
        # are indistinguishable from a genuinely empty band.
        self.ltas_proj = (
            nn.Sequential(
                nn.Conv1d(2, 4 * ltas_channels, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(4 * ltas_channels, ltas_channels, 5, padding=2),
            )
            if ltas_channels
            else None
        )
        self.in_proj = nn.Linear((3 if use_state else 2) + ltas_channels, dim)
        self.freq_embed = nn.Parameter(torch.randn(1, num_bins, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                FreqTransformerBlock(dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.bin_proj = nn.Linear(dim, bin_out)
        self.out_proj = nn.Linear(num_bins * bin_out, out_dim)

    def forward(
        self,
        xt: Optional[Tensor],
        condition: Tensor,
        band_mask: Tensor,
        ltas: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
          xt: the noisy state, of shape (B, T, F), or None when the encoder was
            built with ``use_state=False``.
          condition: the clean band-limited condition, of shape (B, T, F).
          band_mask: how missing each bin is, 1.0 for fully missing, 0.0 for
            fully observed and a ramp in between, of shape (B, 1, F) or
            (B, T, F). This is the soft mask ``band_limit`` returns, kept soft
            so the encoder can see that a bin sits on the transition slope
            rather than squarely on one side of it.
          ltas: the prompt's long-term average spectrum, of shape (B, F),
            averaged over prompt frames only and zero on unobserved bins.
            Required when the encoder was built with ``ltas_channels``.
        Returns:
          A per-frame embedding of shape (B, T, out_dim).
        """
        assert (xt is not None) == self.use_state, (xt is None, self.use_state)
        b, t, f = condition.shape
        assert f == self.num_bins, (f, self.num_bins)
        observed = (1.0 - band_mask.to(condition.dtype)).expand(b, t, f)
        parts = [xt, condition, observed] if self.use_state else [condition, observed]
        if self.ltas_proj is not None:
            assert ltas is not None, "encoder was built with ltas_channels"
            ltas_in = torch.stack([ltas.to(condition.dtype), observed[:, 0]], dim=1)
            # (B, C, F) -> (B, 1, F, C), broadcast over time: the spectrum is
            # constant along the utterance by construction.
            ltas_out = self.ltas_proj(ltas_in).permute(0, 2, 1).unsqueeze(1)
            parts = parts + list(
                ltas_out.expand(b, t, f, self.ltas_channels).unbind(-1)
            )
        tokens = torch.stack(parts, dim=-1)
        x = self.in_proj(tokens).reshape(b * t, f, -1)
        x = x + self.freq_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.bin_proj(x).reshape(b, t, -1)
        return self.out_proj(x)


class FrequencyDecoder(nn.Module):
    """Mirror of :class:`FrequencyEncoder` on the read-out side.

    A per-frame vector from the temporal backbone is broadcast over the
    frequency axis, given learned per-bin embeddings, passed through
    frequency-axis attention, and read out to one velocity value per bin by a
    single shared linear layer. Where the backbone's own ``out_proj`` predicts
    all bins from one matrix, this predicts each bin from a representation that
    has attended to the others.
    """

    def __init__(
        self,
        num_bins: int,
        in_dim: int,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.in_proj = nn.Linear(in_dim, dim)
        self.freq_embed = nn.Parameter(torch.randn(1, num_bins, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                FreqTransformerBlock(dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 1)
        # Initialised small rather than at zero, for the reason given in
        # ``ZipVoiceAnyBand.__init__`` about ScaledAdam: an update is scaled by
        # the tensor's own RMS, so a tensor at the origin receives a zero-sized
        # step. This head was the last zero-initialised matrix in the branch and
        # measured 1.73e-03 after 30k steps against 4.26e-02 for the feed-forward
        # matrix one layer above it -- it escaped the origin through its bias but
        # stayed an order of magnitude below everything trained normally. The
        # step-0 identity property comes from the gate's warm-up, which scales
        # this head's whole contribution in from zero, so nothing is lost by
        # letting the matrix itself start where the optimizer can move it.
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: Tensor, extra: Optional[Tensor] = None) -> Tensor:
        """hidden: (B, T, in_dim) -> velocity residual (B, T, num_bins, 1).

        ``extra`` is an optional term of the same width summed into ``hidden``
        before the input projection. It carries the frequency encoder's output
        when the branch is configured to enter the backbone here rather than at
        its input; see ``ZipVoiceAnyBand`` on ``branch_entry``.
        """
        if extra is not None:
            hidden = hidden + extra
        b, t, _ = hidden.shape
        x = self.in_proj(hidden).reshape(b * t, 1, -1).expand(-1, self.num_bins, -1)
        x = x + self.freq_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.out_proj(x).reshape(b, t, self.num_bins, 1)


class BandAdapter(nn.Module):
    """A ControlNet-style side branch that predicts a per-bin velocity residual.

    :class:`FrequencyEncoder` is the feature extractor: it turns the
    band-limited mel and the mask saying which of its bins are observed into a
    sequence at backbone width. That sequence passes through a lightweight
    temporal encoder -- two 1-D convolutions with SiLU between them, which give
    it a receptive field over neighbouring frames that the per-frame projection
    it comes from does not have -- and becomes the key and value of
    ``num_layers`` cross-attention blocks whose query is the backbone's own
    hidden output. Each block's contribution is projected by a 1x1 convolution
    and summed back into the running state, so the branch reads the backbone
    rather than displacing it. A final linear reads the state out to one
    velocity value per mel bin.

    The 1x1 projections are ControlNet's zero convolutions in role but not in
    initialisation. ControlNet trains under AdamW, where a tensor at the origin
    still receives a full-sized step. This model trains under ScaledAdam, which
    scales every step by the tensor's own per-row rms and suppresses the
    multiplicative scale update below ``param_min_rms`` = 1e-5, so a row that
    starts at exactly zero can never leave it. That is not a hypothetical: an
    earlier version of this branch put a zero-initialised matrix at its entrance
    and measured rms 6.15e-06 at 1k iterations and 5.95e-06 at 9k, shrinking
    under weight decay rather than learning, with everything upstream of it
    decaying alongside because their gradient arrived through it. 1e-3 is small
    enough that the branch starts as a negligible perturbation and large enough
    that the optimizer can scale it. The identity-at-full-band property that a
    true zero convolution would have bought at step 0 only is supplied instead,
    and permanently, by the caller: ``ZipVoiceAdapter`` multiplies this output
    by the band mask, which is identically zero when nothing is missing.
    """

    def __init__(
        self,
        dim: int,
        out_bins: int,
        query_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        """``dim`` is the branch's own width and ``query_dim`` the backbone's.

        They are separated because the branch does not need the backbone's 512
        channels to say which bins are missing and how loud they should be: run
        at that width its two 5-tap convolutions alone cost 2.6M parameters,
        against 462k for the entire previous design, on a corpus this is
        fine-tuned on for ten thousand steps. A linear projection brings the
        backbone's hidden state down to ``dim`` at the query, and the read-out
        goes straight from ``dim`` to mel bins, so the wide tensors appear only
        at the two boundaries.
        """
        super().__init__()
        pad = kernel_size // 2
        query_dim = dim if query_dim is None else query_dim
        self.q_proj = nn.Linear(query_dim, dim) if query_dim != dim else nn.Identity()
        self.encoder = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=pad),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size, padding=pad),
            nn.SiLU(),
        )
        self.q_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.kv_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    dim, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.out_convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, 1) for _ in range(num_layers)]
        )
        for conv in self.out_convs:
            nn.init.normal_(conv.weight, std=1e-3)
            nn.init.zeros_(conv.bias)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_bins)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden: Tensor,
        cond: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """hidden: (B, T, query_dim), cond: (B, T, dim) -> (B, T, out_bins).

        ``padding_mask`` is True at padded frames and keeps them out of the
        attention's keys, where they would otherwise contribute a learned bias
        to every real frame's output.
        """
        c = self.encoder(cond.transpose(1, 2)).transpose(1, 2)
        state = self.q_proj(hidden)
        for q_norm, kv_norm, attn, conv in zip(
            self.q_norm, self.kv_norm, self.attn, self.out_convs
        ):
            delta, _ = attn(
                q_norm(state),
                kv_norm(c),
                kv_norm(c),
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            state = state + conv(delta.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(self.norm(state))


class F0Embedding(nn.Module):
    """Project a frame-level F0 and voicing contour to the conditioning width.

    AnyBand adds this to the frequency encoder's output rather than
    concatenating it, which is what is done here.
    """

    def __init__(self, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, f0_norm: Tensor, voiced: Tensor) -> Tensor:
        """f0_norm, voiced: (B, T) -> (B, T, out_dim)."""
        x = torch.stack([f0_norm, voiced.to(f0_norm.dtype)], dim=-1)
        return self.net(x)


class MelF0Estimator(nn.Module):
    """Predict a normalised F0 contour and a voicing flag from a mel.

    This stands in for pYIN, which needs a waveform that does not exist over the
    span a text-to-speech model is generating. It is trained on its own against
    pYIN targets and then frozen, so it behaves as a fixed feature extractor and
    contributes no gradient to the flow-matching model.

    The architecture is deliberately small -- a dilated convolutional stack over
    time -- because F0 is a low-dimensional, strongly local quantity and the
    module has to run inside the ODE loop.
    """

    def __init__(
        self,
        num_bins: int,
        channels: int = 192,
        num_layers: int = 5,
        kernel_size: int = 5,
    ):
        super().__init__()
        layers = []
        in_ch = num_bins
        for i in range(num_layers):
            dilation = 2**i
            padding = dilation * (kernel_size - 1) // 2
            layers.append(
                nn.Conv1d(
                    in_ch,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            layers.append(nn.GroupNorm(8, channels))
            layers.append(nn.SiLU())
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 2, 1)

    def forward(self, mel: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          mel: (B, T, F).
        Returns:
          ``(f0_norm, voiced_logit)``, both of shape (B, T). ``f0_norm`` is in
          [0, 1] on the same log scale as :func:`normalize_f0`; ``voiced_logit``
          is a raw logit, so a caller that wants a hard flag applies its own
          threshold.
        """
        x = self.net(mel.transpose(1, 2))
        out = self.head(x).transpose(1, 2)
        return torch.sigmoid(out[..., 0]), out[..., 1]


class TimestepGate(nn.Module):
    """A single monotone gate over the flow-matching timestep.

    The base conditioning is weighted ``1 - alpha(t)`` and the frequency
    conditioning ``alpha(t)``, with one shared alpha rather than two independent
    gates. Two gates would admit the solution "base on, frequency off", which is
    precisely the collapse an earlier conditioning experiment in this repository
    was measured to fall into; tying them makes ignoring the new path cost the
    model its old one.

    Monotonicity is structural, not a penalty: the pre-activation is a
    non-negative combination of monotone increasing basis functions -- the
    identity and a set of hinges -- so it can only increase in t, and the
    sigmoid preserves that. This encodes the intended reading of the schedule,
    that semantic content is settled early from text and the frequency detail is
    added late, without fixing where the transitions fall.

    The initialisation reproduces the hand-designed 8/4/4 schedule of a 16-step
    solver: alpha is near 0 over the first half, crosses 0.5 at t = 0.6, and is
    near 1 over the last quarter. ``warmup_steps`` then scales the whole gate in
    from zero, so that at step 0 the model is bit-for-bit the pretrained one and
    the schedule is reached over the first few thousand updates rather than
    imposed as a shock.

    ``max_alpha`` is strictly below 1 so the base conditioning is never switched
    off completely. Classifier-free guidance operates by zeroing that base
    conditioning, so a gate that reached exactly 1 would make guidance a no-op at
    the end of the trajectory.

    ``out_bins`` selects between the two gates this model instantiates. At 1 the
    gate is the scalar alpha that scales the encoder's contribution into the
    backbone's residual stream, where a per-bin quantity would be meaningless:
    ``extra_in`` lives at backbone width and its coordinates are not mel bins.
    At ``num_bins`` it is the read-out gate, whose contribution *is* in mel-bin
    space, so each bin can be opened independently. That matters because the
    read-out addition is the one place the branch can damage bins the prompt
    already observes: with a single scalar, any gain above the cutoff is paid
    for with a disturbance below it, and speaker similarity reads that
    disturbance. Only the ceiling is widened, not the schedule -- how far a bin
    may open is a property of the bin, while when the branch opens along the
    trajectory is a property of the trajectory and stays shared.

    ``cond_dim`` is the width of the conditioning vector read alongside the
    band fraction; it carries the prompt's long-term average spectrum, so the
    gate adapts to the voice and the recording chain rather than to bandwidth
    alone.
    """

    def __init__(
        self,
        num_knots: int = 8,
        warmup_steps: int = 2000,
        max_alpha: float = 0.98,
        band_hidden_dim: int = 32,
        out_bins: int = 1,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.out_bins = out_bins
        self.cond_dim = cond_dim
        # A slope of 10.7 with a bias of -6.42 puts alpha at 0.02 / 0.5 / 0.95
        # at t = 0.25 / 0.6 / 0.9. softplus is very nearly the identity this far
        # out, so the raw parameter is the slope itself to five decimal places.
        self.slope = nn.Parameter(torch.tensor(10.7))
        # softplus(-10) ~= 4.5e-05, so the hinges start inactive and the gate
        # starts as a plain logistic ramp.
        self.hinge_weight = nn.Parameter(torch.full((num_knots,), -10.0))
        self.register_buffer(
            "knots", torch.linspace(0.05, 0.95, num_knots).unsqueeze(0)
        )
        self.bias = nn.Parameter(torch.tensor(-6.42))
        # Bandwidth conditioning, as one block with the timestep rather than a
        # second additive term. An additive band term can only shift the logit,
        # which slides the same schedule earlier or later; it cannot change its
        # shape. That is the wrong degree of freedom. A prompt cut at 4 kHz has
        # to invent most of the spectrum and wants the branch early and hard; a
        # prompt cut at 16 kHz wants a late, shallow touch. Those are different
        # curves, not one curve at two offsets.
        #
        # So the missing-band fraction conditions the schedule's own parameters
        # -- slope, bias and every hinge weight -- and the gate is evaluated
        # once from the pair. Monotonicity in t survives unchanged: softplus
        # keeps the slope and all hinge weights positive whatever the band asks
        # for, so alpha can still only increase along the trajectory.
        #
        # The read-out is initialised small rather than at exactly zero. At zero
        # the schedule would start as the tuned timestep-only ramp, which is
        # what we want, but ScaledAdam scales each tensor's update by that
        # tensor's own RMS, so a zero tensor receives a zero-sized step and
        # never leaves the origin. 1e-3 is negligible against a slope of 10.7
        # and still gives the optimizer something to scale.
        self.band_mlp = nn.Sequential(
            nn.Linear(1 + cond_dim, band_hidden_dim),
            nn.SiLU(),
            nn.Linear(band_hidden_dim, num_knots + 2 + out_bins),
        )
        nn.init.normal_(self.band_mlp[-1].weight, std=1e-3)
        nn.init.normal_(self.band_mlp[-1].bias, std=1e-3)
        # The block's third read-out is the ceiling: how far the branch may open
        # at this bandwidth, as opposed to how early. Without it the bandwidth
        # can only move the schedule along t, and once the logistic saturates --
        # which a slope near 10 reaches by t = 0.3 -- its derivative is ~3e-4
        # and no bandwidth gradient survives. Measured on a 18k-step run: the
        # spread of alpha across band_frac was 0.642 at t = 0, 0.029 at t = 0.3
        # and 0.000 from t = 0.5 on, so over most of the trajectory the gate was
        # a constant 0.98 whatever the prompt's bandwidth. The ceiling is a
        # factor outside the timestep logistic, so it keeps its gradient exactly
        # where that logistic loses its own.
        #
        # ``ceil_bias`` of 3.0 puts the ceiling at sigmoid(3) = 0.953 of
        # ``max_alpha`` at initialisation, close to the tuned schedule while
        # leaving the sigmoid a derivative of 0.045 to learn through. Starting
        # it high enough to be exactly 1.0 would reintroduce the saturation this
        # exists to avoid.
        self.ceil_bias = nn.Parameter(torch.full((out_bins,), 3.0))
        self.warmup_steps = warmup_steps
        self.max_alpha = max_alpha
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def warmup_scale(self) -> float:
        # Warm-up is a training-time device: it eases the gate in from zero over
        # the first few thousand updates so the pretrained model is not shocked.
        # At inference there is nothing to ease in, and the counter it reads is
        # not reliable there -- ``step`` is an integer buffer, and
        # ``average_state_dict`` copies only floating-point tensors, so every
        # averaged checkpoint carries the freshly-constructed 0. Reading it in
        # eval mode therefore scaled the whole frequency branch to zero, which
        # is how the first round of AnyBand results came to be stock with a dead
        # branch bolted on. Evaluation always uses the full gate.
        if not self.training:
            return 1.0
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.step.item()) / self.warmup_steps)

    def forward(
        self,
        t: Tensor,
        band_frac: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate the gate.

        Args:
            t: (B,) or (B, T).
            band_frac: the fraction of mel bins absent from the prompt, of the
                same shape as ``t``, in [0, 1]. This is the bandwidth the gate
                conditions on; it is read off the band mask rather than taken as
                a sampling rate so that a single quantity, already threaded to
                every call site for both training and inference, carries it.
            cond: an optional (B, cond_dim) conditioning vector, constant over
                the trajectory, broadcast to every entry of ``t``.

        Returns:
            alpha, of shape ``t.shape`` when ``out_bins`` is 1 and
            ``t.shape + (out_bins,)`` otherwise.
        """
        shape = t.shape
        flat = t.reshape(-1, 1)
        if band_frac is None:
            band_in = torch.zeros_like(flat)
        else:
            band_in = band_frac.reshape(-1, 1).to(dtype=flat.dtype)

        if self.cond_dim:
            assert cond is not None, "gate was built with cond_dim but got none"
            # ``cond`` is per utterance while ``flat`` is per (utterance, frame)
            # entry, so it is repeated rather than reshaped.
            cond = cond.to(dtype=flat.dtype)
            reps = flat.size(0) // cond.size(0)
            band_in = torch.cat([band_in, cond.repeat_interleave(reps, 0)], dim=1)

        # One evaluation of the joint block: the band fraction picks a schedule,
        # then t is read off it.
        mod = self.band_mlp(band_in)
        n = self.out_bins
        slope = F.softplus(self.slope + mod[:, 0])
        bias = self.bias + mod[:, 1]
        ceiling = torch.sigmoid(self.ceil_bias.unsqueeze(0) + mod[:, 2 : 2 + n])
        hinge_weight = F.softplus(
            self.hinge_weight.unsqueeze(0) + mod[:, 2 + n :]
        )

        hinges = F.relu(flat - self.knots)
        logit = (
            slope * flat.squeeze(-1)
            + (hinges * hinge_weight).sum(dim=-1)
            + bias
        )
        # Monotone in t as before: the ceiling does not depend on t, so scaling
        # by it cannot make a rising curve fall. With a per-bin ceiling every
        # bin gets its own height on one shared rising curve.
        alpha = torch.sigmoid(logit).unsqueeze(-1) * ceiling * self.max_alpha
        alpha = alpha * self.warmup_scale()
        return alpha.reshape(shape) if n == 1 else alpha.reshape(*shape, n)
