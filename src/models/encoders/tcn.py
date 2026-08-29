"""
v1 encoder (README Phase 3): dilated causal TCN. 8 blocks, dilations
1,2,4,...,128, kernel_size=2, single causal conv per block -> receptive
field = 1 + sum(dilations) = 1 + 255 = 256 bars, matching the README spec
exactly.

Causality is structural, not policy: each block pads ONLY on the left
(the past) before convolving, so output position t is a function of inputs
at positions <= t only, never t+1 onward. test_tcn.py checks this directly
by perturbing a future timestep and asserting the output at t is unchanged
-- the same kind of structural guarantee the LSTM gets for free from being
recurrent, made explicit here since a conv net doesn't get it for free.

Unlike the LSTM (sequential over time -> no parallelism across timesteps,
which is exactly what made M2's baseline run take 6.5 hours on CPU),
convolutions are parallel across the time dimension, so this should be
genuinely fast on CPU, not just nominally so.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TCN_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]  # 8 blocks -> receptive field 256 w/ kernel_size=2
TCN_CHANNELS = 64
TCN_KERNEL_SIZE = 2


class CausalTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, seq_len] -> [batch, out_channels, seq_len] (same length, causal)."""
        out = F.pad(x, (self.left_pad, 0))  # pad the PAST side only
        out = self.conv(out)
        out = self.dropout(self.relu(out))
        return self.relu(out + self.residual_proj(x))


class TCNEncoder(nn.Module):
    """Stack of causal dilated blocks. Returns the full sequence of
    representations [batch, seq_len, channels] -- callers (a regression head,
    an SSL head, ...) decide whether they want the last timestep or all of
    them (e.g. masked patch reconstruction needs every position)."""

    def __init__(self, n_features: int, channels: int = TCN_CHANNELS,
                 dilations: list[int] = TCN_DILATIONS, kernel_size: int = TCN_KERNEL_SIZE,
                 dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_ch = n_features
        for d in dilations:
            blocks.append(CausalTCNBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.blocks = nn.ModuleList(blocks)
        self.receptive_field = 1 + sum(dilations) * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, n_features] -> [batch, seq_len, channels]."""
        out = x.transpose(1, 2)  # -> [batch, features, seq_len]
        for block in self.blocks:
            out = block(out)
        return out.transpose(1, 2)  # -> [batch, seq_len, channels]


class TCNBaseline(nn.Module):
    """Same drop-in interface as LSTMBaseline: [batch, seq_len, features] -> [batch] scalar."""

    def __init__(self, n_features: int, channels: int = TCN_CHANNELS, dropout: float = 0.1):
        super().__init__()
        self.encoder = TCNEncoder(n_features, channels=channels, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out = self.encoder(x)          # [batch, seq_len, channels]
        last = seq_out[:, -1, :]           # causal: last position summarizes all of x
        return self.head(last).squeeze(-1)
