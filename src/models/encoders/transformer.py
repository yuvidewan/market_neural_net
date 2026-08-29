"""
v2 encoder (README Phase 3): the two-axis causal Transformer -- "the main
model". Unlike the LSTM/TCN (which process one symbol's window in complete
isolation, batch dimension = unrelated independent samples), this encoder's
whole point is a SECOND attention axis across symbols, so it structurally
CANNOT be called one symbol at a time -- a forward pass takes an entire
cross-sectional panel (all symbols as of the same date) at once. See
src/data/panel_dataset.py for the dataset that supplies panels shaped this
way, and scripts/train_transformer_ssl.py for how the two connect.

Architecture, per README Phase 3 v2 spec:
  1. Patchify: `patch_size` consecutive bars -> one token (PatchTST-style).
     Left-padded (the PAST side) if seq_len isn't divisible by patch_size,
     same causal convention as the TCN's left-padding -- the most recent
     bar always lands in the last patch, never diluted by padding.
  2. N blocks, each alternating:
     a. TEMPORAL attention -- causal, RoPE, WITHIN one symbol (attention
        axis = patches/time; batch axis = symbols). A patch's representation
        may depend on itself and earlier patches, never a later one.
     b. CROSS-SECTIONAL attention -- unmasked, WITHIN one patch position
        (attention axis = symbols; batch axis = patches). This is the axis
        that can learn sector rotation / lead-lag / index effects without
        anyone telling it what a sector is -- and it does NOT touch the time
        axis, so it cannot introduce lookahead: mixing across symbols at a
        FIXED patch position never touches a later position for any symbol.
     c. Feedforward.
  3. Output: [n_symbols, n_patches, d_model] -- same [batch, seq, hidden]
     shape convention the LSTM/TCN encoders use (here "batch" = symbols,
     "seq" = patches), so EncoderQuantileWrapper (src/models/ssl/quantile.py)
     plugs in completely unchanged: `seq_out[:, -1, :]` is still "the causal
     summary as of the most recent data", just per-symbol-in-a-panel instead
     of per-independent-sample.

Causality is structural, not policy, exactly like the TCN: temporal
attention's `is_causal=True` mask makes it a hard constraint of the
attention computation itself, not a training convention that could be
silently violated. test_transformer.py checks this directly (perturbing a
future patch leaves earlier outputs bit-for-bit unchanged) across the full
multi-block stack, not just one block -- residual/reshape bugs that
accidentally leak across the symbol<->patch transpose would show up there.

KNOWN SIMPLIFICATION, stated rather than hidden: patches are aligned by
ORDINAL position within each symbol's own trailing window (patch N-1 = each
symbol's own most recent `patch_size` rows), not by calendar date. If a
symbol has an occasional missing session inside the window, its patch
boundaries drift by that many rows relative to a symbol with no gaps. This
never breaks the causality guarantee (every symbol's window still ends
exactly at the panel's anchor date, never later) -- it's the same
trailing-N-ROWS-not-trailing-N-CALENDAR-DAYS convention every other part of
this pipeline already uses (SequenceDataset, the TCN's receptive field).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TRANSFORMER_D_MODEL = 256
TRANSFORMER_N_HEADS = 8
TRANSFORMER_N_BLOCKS = 8
TRANSFORMER_PATCH_SIZE = 16


def _build_rope_cache(n_positions: int, head_dim: int, device, dtype, base: float = 10000.0):
    """cos/sin tables for rotary position embedding, one entry per position
    per pair-of-dims. head_dim must be even (pairs get rotated together)."""
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(n_positions, device=device, dtype=dtype)
    freqs = torch.outer(positions, theta)  # [n_positions, head_dim/2]
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., n_positions, head_dim]. Rotates each adjacent (even, odd)
    pair of dims by the position-dependent angle -- standard RoPE."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class PatchEmbed(nn.Module):
    def __init__(self, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        self.proj = nn.Linear(patch_size * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_symbols, seq_len, n_features] -> [n_symbols, n_patches, d_model]."""
        n_symbols, seq_len, n_features = x.shape
        remainder = seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, 0, pad_len, 0))  # left-pad the PAST side only
            seq_len = x.shape[1]
        n_patches = seq_len // self.patch_size
        x = x.reshape(n_symbols, n_patches, self.patch_size * n_features)
        return self.proj(x)


class TemporalAttention(nn.Module):
    """Causal self-attention across patches, independently per symbol."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_symbols, n_patches, d_model] -> same shape."""
        n_symbols, n_patches, d_model = x.shape
        qkv = self.qkv(x).reshape(n_symbols, n_patches, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [n_symbols, n_patches, n_heads, head_dim]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # -> [n_symbols, n_heads, n_patches, head_dim]

        cos, sin = _build_rope_cache(n_patches, self.head_dim, x.device, x.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )  # [n_symbols, n_heads, n_patches, head_dim] -- causal: position i attends only to j<=i
        attn_out = attn_out.transpose(1, 2).reshape(n_symbols, n_patches, d_model)
        return self.out_proj(attn_out)


class CrossSectionalAttention(nn.Module):
    """Unmasked self-attention across symbols, independently per patch
    position. No RoPE -- symbols form an unordered set, not a sequence."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_symbols, n_patches, d_model] -> same shape."""
        n_symbols, n_patches, d_model = x.shape
        x_t = x.transpose(0, 1)  # [n_patches, n_symbols, d_model] -- attend across symbols per patch
        qkv = self.qkv(x_t).reshape(n_patches, n_symbols, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # -> [n_patches, n_heads, n_symbols, head_dim]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout if self.training else 0.0
        )  # unmasked: every symbol sees every symbol AT THIS SAME PATCH POSITION only
        attn_out = attn_out.transpose(1, 2).reshape(n_patches, n_symbols, d_model)
        return self.out_proj(attn_out).transpose(0, 1)  # back to [n_symbols, n_patches, d_model]


class TwoAxisBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.temporal = TemporalAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_sectional = CrossSectionalAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal(self.ln1(x))
        x = x + self.cross_sectional(self.ln2(x))
        x = x + self.ff(self.ln3(x))
        return x


class TwoAxisTransformerEncoder(nn.Module):
    """
    [n_symbols, seq_len, n_features] -> [n_symbols, n_patches, d_model].
    Same [batch, seq, hidden] convention as LSTMEncoder/TCNEncoder (here
    "batch"=symbols, "seq"=patches) -- callers (EncoderQuantileWrapper, a
    future policy head) treat it identically regardless of which encoder
    produced it, EXCEPT that a forward() call here must be given an entire
    cross-sectional panel, never one symbol alone (see module docstring).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = TRANSFORMER_D_MODEL,
        n_heads: int = TRANSFORMER_N_HEADS,
        n_blocks: int = TRANSFORMER_N_BLOCKS,
        patch_size: int = TRANSFORMER_PATCH_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(n_features, patch_size, d_model)
        self.blocks = nn.ModuleList([TwoAxisBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_symbols, seq_len, n_features] -- an ENTIRE panel, one date."""
        out = self.patch_embed(x)
        for block in self.blocks:
            out = block(out)
        return self.ln_out(out)
