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

"""Attention from one model's state to another model's state.

Zipformer's own attention (``RelPositionMultiheadAttentionWeights`` together
with ``SelfAttention``) derives its weights from the sequence it is attending
over and reuses them across several submodules of a layer, which is what makes
that layer cheap. It cannot take keys from somewhere else. This module is the
small piece that can: query from the running state, key and value from an
external memory.

The memory here is the frozen backbone's hidden state at the current solver
step, so the keys are recomputed every step rather than carried. That memory is
on the same frame grid as the query -- both models run at the feature rate --
so no resampling is needed and the attention is free to look at neighbouring
frames rather than only at the aligned one.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttention(nn.Module):
    """Multi-head attention from ``src`` to an external ``memory``.

    Shapes follow Zipformer's convention for the running state, which is
    time-first: ``src`` is (seq_len, batch, embed_dim). ``memory`` is
    batch-first, (batch, seq_len, memory_dim), because that is how the backbone
    hands its hidden state back.

    ``out_proj`` starts small rather than at zero. The branch is trained under
    ScaledAdam, which scales each step by the tensor's own per-row rms and
    suppresses the multiplicative update below ``param_min_rms`` = 1e-5, so a
    row initialised at exactly zero can never leave it -- measured on an earlier
    branch as rms 6.15e-06 at 1k iterations and 5.95e-06 at 9k, shrinking under
    weight decay instead of learning. 1e-3 is small enough that the branch
    begins as though blind to the backbone and large enough that the optimizer
    can grow it.
    """

    def __init__(
        self,
        embed_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (embed_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(memory_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        src: Tensor,
        memory: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """src: (T, B, C), memory: (B, S, M) -> the residual to add, (T, B, C).

        ``key_padding_mask`` is (B, S), True at padded frames; those frames are
        kept out of the keys, where they would otherwise contribute a learned
        bias to every real frame.
        """
        seq_len, batch, _ = src.shape

        q = self.q_proj(self.norm_q(src))
        q = q.reshape(seq_len, batch, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3)  # (B, H, T, D)

        mem = self.norm_kv(memory)
        k = self.k_proj(mem).reshape(batch, -1, self.num_heads, self.head_dim)
        v = self.v_proj(mem).reshape(batch, -1, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (B, H, S, D)
        v = v.transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.all(dim=1).any():
                # An utterance whose keys are entirely padded would give a row
                # of -inf and a NaN after the softmax. It cannot happen for a
                # real batch, where every utterance has at least one frame, so
                # assert rather than paper over it.
                raise ValueError("cross-attention memory is fully padded")
            attn_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
            attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, S)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.permute(2, 0, 1, 3).reshape(seq_len, batch, -1)
        return self.out_proj(out)
