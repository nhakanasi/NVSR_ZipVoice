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

"""Conditioning streams for a bandwidth-extension side branch.

A stream turns some view of the band-limited prompt into a per-frame sequence.
Streams are concatenated along the channel axis, so each keeps its own channels
rather than having to share them, and the branch that consumes them --
:class:`~zipvoice.models.zipvoice_dual.HighBandBranch` -- projects the result
back to its own width at the entrance.

These three encoders were measured on the earlier ControlNet-style branch (see
the ``adapter-arm`` branch for the model they fed and for its numbers); they are
carried over here unchanged so the two arms are conditioned identically and the
comparison is about the generative objective rather than about what the model
was told.

:class:`FrequencyEncoder` is the first stream and the only mandatory one. It is
AnyBand's frequency encoder ("AnyBand: Unified Multi-Bandwidth Speech Extension
via Frequency-Aware In-Context Spectral Infilling", arXiv 2608.00572). Its
observation is that a frame-wise projection of a mel spectrogram destroys the
structure that decides what belongs in a missing band: the relationship
*between* bins. So the mel bins of a single frame are treated as a token
sequence and self-attention runs along the frequency axis, independently per
frame, letting an observed low band speak directly to the bins that have to be
invented above it.

:class:`SpeakerEncoder` is an optional stream carrying the prompt's time-pooled
spectral statistics, and :class:`RateEncoder` an optional stream carrying the
cutoff frequency as a number rather than only as the shape of a mask.

A stream carrying a pitch track was tried and removed. It cost 0.087 to 0.231
SIM-o at every cutoff against a 0.008 noise floor, for no UTMOS gain, and the
training objective could not see it: validation was identical on all three
terms. Pitch is cheap, reliable and *not identifying*, so conditioned on it the
branch can write the conditional mean over every speaker at that pitch, which
both loss terms accept -- both are minimised by a conditional mean -- and which
is by construction the least speaker-specific band available. The lesson is in
what is admitted here now: a stream must carry speaker identity, or it hands
the branch a way to satisfy the objective without carrying any.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from zipvoice.models.modules.zipformer import timestep_embedding


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
    """AnyBand's frequency encoder, used here as the branch's feature extractor.

    Every mel bin of every frame is a token carrying two values: the bin's value
    in the clean band-limited condition, and how far the condition is observed
    there. Without that second value a missing bin and a genuinely silent one
    are indistinguishable, since both sit at the log-mel floor. Attention along
    the frequency axis then lets the observed bins inform the missing ones
    within that frame.

    The noisy flow-matching state is deliberately not an input, although
    AnyBand's encoder takes it. Here the state reaches the branch through the
    backbone's hidden output, which :class:`BandAdapter` cross-attends to, so
    reading it again would only duplicate it.

    The frequency axis is folded into the channel axis rather than pooled away.
    An earlier version ended in learned-query attention pooling, which collapsed
    all ``num_bins`` tokens to a single vector: every bin's identity was
    destroyed at exactly the point the branch was supposed to be delivering
    per-bin detail, leaving only a per-frame summary. Here each token is instead
    read out to ``bin_out`` values, the result is flattened over frequency, and
    one linear layer maps ``num_bins * bin_out`` to ``out_dim``. Bin identity
    survives as position in the flattened vector, which is the same way the
    backbone's own ``in_proj`` consumes a mel. Because the blocks above have
    already given every token global context along frequency, no separate pooled
    summary is needed.

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
        out_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.in_proj = nn.Linear(2, dim)
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

    def forward(self, condition: Tensor, band_mask: Tensor) -> Tensor:
        """
        Args:
          condition: the clean band-limited condition, of shape (B, T, F).
          band_mask: how missing each bin is, 1.0 for fully missing, 0.0 for
            fully observed and a ramp in between, of shape (B, 1, F) or
            (B, T, F). This is the soft mask ``band_limit`` returns, kept soft
            so the encoder can see that a bin sits on the transition slope
            rather than squarely on one side of it.
        Returns:
          A per-frame embedding of shape (B, T, out_dim).
        """
        b, t, f = condition.shape
        assert f == self.num_bins, (f, self.num_bins)
        observed = (1.0 - band_mask.to(condition.dtype)).expand(b, t, f)
        tokens = torch.stack([condition, observed], dim=-1)
        x = self.in_proj(tokens).reshape(b * t, f, -1)
        x = x + self.freq_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.bin_proj(x).reshape(b, t, -1)
        return self.out_proj(x)


class SpeakerEncoder(nn.Module):
    """The prompt's time-pooled spectral statistics, broadcast over time.

    This is the stream that replaced the pitch track, and the reason for the
    swap is the module docstring's: a conditioning stream that is informative
    but not identifying lets the branch satisfy the objective with a conditional
    mean, which is exactly the band that scores worst on speaker similarity.
    The statistics here are identifying by construction -- they are the
    speaker's own average spectrum and its variation -- so using them *is*
    carrying identity rather than a way around it.

    Per mel bin, over the prompt frames only: the mean and the standard
    deviation of the band-limited condition. The mean is the long-term average
    spectrum, which carries the speaker's timbre and the recording chain's
    signature together; the deviation says how much each bin moves, separating
    a bin held at a steady level from one that swings. One further channel says
    whether the prompt was non-empty at all, since ``condition_time_mask`` draws
    it as 0-30% of the utterance and it is regularly empty, and then the
    statistics are zero for a reason the branch could not otherwise distinguish
    from a silent prompt.

    The branch's cross-attention already pools over the utterance and could in
    principle form these itself. It has not had to: two attention layers whose
    queries come from the backbone are a poor place to compute a masked mean
    over several hundred frames, and the measured result is a branch that loses
    speaker similarity at every bandwidth. Computing it directly costs one
    reduction.

    Nothing here is a leak. The statistics are taken from the band-limited
    condition, the same tensor the frequency encoder reads, so they describe
    only the observed band; the high band remains something the branch must
    infer from the speaker's low band rather than something it is shown.
    """

    def __init__(self, num_bins: int, out_dim: int = 128, dim: int = 256):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(2 * num_bins + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, condition: Tensor, prompt_mask: Tensor) -> Tensor:
        """
        Args:
          condition: the band-limited, prompt-masked condition mel, (B, T, F).
          prompt_mask: True at frames belonging to the prompt, (B, T).
        Returns:
          (B, T, out_dim), constant along time.
        """
        dtype = condition.dtype
        keep = prompt_mask.to(dtype).unsqueeze(-1)
        count = keep.sum(dim=1).clamp(min=1.0)
        mean = (condition * keep).sum(dim=1) / count
        var = ((condition - mean.unsqueeze(1)) ** 2 * keep).sum(dim=1) / count
        has_prompt = (prompt_mask.any(dim=1, keepdim=True)).to(dtype)
        stats = torch.cat([mean, var.clamp(min=1e-8).sqrt(), has_prompt], dim=-1)
        return self.net(stats).unsqueeze(1).expand(-1, condition.size(1), -1)


class RateEncoder(nn.Module):
    """The cutoff frequency itself, broadcast over time.

    Until this stream, the cutoff reached the model only through the *shape* of
    the band mask -- where the ramp sits among the mel bins. That is a coarse
    and indirect encoding of a quantity the branch cares about directly, since
    it says how far the extrapolation has to reach. Here the number is given as
    a number, through the same sinusoidal embedding the backbone uses for its
    own scalar inputs.

    The output is constant along time, so on its own it is a per-utterance bias
    on the cross-attention keys. That is what it is meant to be.
    """

    def __init__(self, out_dim: int = 64, dim: int = 64, scale_hz: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale_hz = scale_hz
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, cutoff_hz: Tensor, num_frames: int) -> Tensor:
        """cutoff_hz: (B,) in Hz -> (B, num_frames, out_dim)."""
        emb = timestep_embedding(cutoff_hz.float() / self.scale_hz, self.dim)
        return self.out_proj(emb).unsqueeze(1).expand(-1, num_frames, -1)
