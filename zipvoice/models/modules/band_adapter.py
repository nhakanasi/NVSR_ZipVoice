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

"""The side branch of :class:`~zipvoice.models.zipvoice_adapter.ZipVoiceAdapter`.

Two modules, in the order the branch uses them.

:class:`FrequencyEncoder` is the feature extractor, and is AnyBand's frequency
encoder ("AnyBand: Unified Multi-Bandwidth Speech Extension via Frequency-Aware
In-Context Spectral Infilling", arXiv 2608.00572). Its observation is that a
frame-wise projection of a mel spectrogram destroys the structure that decides
what belongs in a missing band: the relationship *between* bins. So the mel bins
of a single frame are treated as a token sequence and self-attention runs along
the frequency axis, independently per frame, letting an observed low band speak
directly to the bins that have to be invented above it.

:class:`BandAdapter` is the ControlNet-style branch proper: the encoder's output
is cross-attended by the frozen backbone's own hidden state and read out to one
velocity residual per mel bin.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class BandAdapter(nn.Module):
    """A ControlNet-style side branch that predicts a per-bin velocity residual.

    :class:`FrequencyEncoder` is the feature extractor: it turns the
    band-limited mel and the mask saying which of its bins are observed into a
    per-frame sequence. That sequence passes through a lightweight temporal
    encoder -- two 1-D convolutions with SiLU between them, which give it a
    receptive field over neighbouring frames that the per-frame projection it
    comes from does not have -- and becomes the key and value of ``num_layers``
    cross-attention blocks whose query is the backbone's own hidden output. Each
    block's contribution is projected by a 1x1 convolution and summed back into
    the running state, so the branch reads the backbone rather than displacing
    it. A final linear reads the state out to one velocity value per mel bin.

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
