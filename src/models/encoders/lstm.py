"""
v0 baseline encoder (README Phase 3): 2-layer LSTM, hidden=128, over a
120-bar causal window. This is the floor every later architecture (TCN,
transformer) must beat -- not the interesting model, the sanity check.

Causality is structural here, not policy: an LSTM only ever sees timesteps
0..t when producing its hidden state at t, by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, n_features] -> [batch] predicted next-bar return."""
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, hidden_size], final layer's final timestep
        return self.head(last_hidden).squeeze(-1)
