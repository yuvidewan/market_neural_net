"""
Self-supervised objective #2 from README Phase 3: "autoregressive next-bar
prediction -- predict the *distribution* of the next return via quantile
regression (pinball loss at 9 quantiles), not a point estimate."

This is genuinely self-supervised in the sense the README means it: the
label (the next bar's realized return) is free, already present in the raw
time series -- no human annotation, no external data source. It's also
encoder-agnostic by design: QuantileHead takes whatever fixed-size
representation an encoder produces (LSTM's last hidden state, TCN's last
timestep, later the transformer's pooled token) and is otherwise identical
regardless of which encoder produced it -- the whole point of Phase 3's
"three encoders, one interface" design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# README spec: 9 quantiles
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class QuantileHead(nn.Module):
    def __init__(self, in_features: int, quantiles: list[float] = QUANTILES, hidden: int | None = None):
        super().__init__()
        self.quantiles = quantiles
        hidden = hidden or max(in_features // 2, len(quantiles))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, in_features] -> [batch, n_quantiles] raw quantile predictions
        (NOT sorted/monotonic-enforced -- see module docstring caveat below)."""
        return self.net(x)

    def median_index(self) -> int:
        return self.quantiles.index(0.5)


class EncoderQuantileWrapper(nn.Module):
    """
    Generic pretraining head: any encoder that maps [batch, seq_len, features]
    -> [batch, seq_len, hidden] (a full per-timestep sequence, not just a
    pooled vector -- TCNEncoder already does this) gets a QuantileHead
    attached to its last (causal) timestep. Works unchanged for the TCN now
    and the transformer later -- the "three encoders, one interface" design
    from README Phase 3 applies to the SSL objective too, not just the
    point-estimate baseline.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, quantiles: list[float] = QUANTILES):
        super().__init__()
        self.encoder = encoder
        self.head = QuantileHead(hidden_size, quantiles=quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out = self.encoder(x)       # [batch, seq_len, hidden]
        last = seq_out[:, -1, :]        # causal: summarizes all of x
        return self.head(last)          # [batch, n_quantiles]


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    preds: [batch, n_quantiles], target: [batch]. Standard quantile
    ("pinball") loss: for quantile tau, error = target - pred; loss is
    tau*error when error>=0 (under-prediction), (tau-1)*error when error<0
    (over-prediction) -- asymmetric by design so tau=0.9's prediction learns
    to sit above 90% of outcomes, not at the mean.
    """
    target = target.unsqueeze(-1)  # [batch, 1] broadcasts against [batch, n_quantiles]
    errors = target - preds
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device).unsqueeze(0)  # [1, n_quantiles]
    loss = torch.maximum((q - 1) * errors, q * errors)
    return loss.mean()


# KNOWN LIMITATION, stated rather than hidden: nothing here enforces
# non-crossing quantiles (predicted q0.1 could come out numerically above
# predicted q0.5 for a given sample, especially early in training). This is
# a well-known real limitation of vanilla independent quantile regression.
# Acceptable for M3's purpose (does the pipeline detect real structure at
# all -- rank IC uses only the median quantile's prediction, see
# scripts/train_ssl_quantile.py), but a monotonicity penalty or an explicit
# cumulative-sum parameterization would be needed before quantile spread is
# trusted as a real confidence signal (the "target/stop-loss from quantiles"
# design in README Phase 5).
