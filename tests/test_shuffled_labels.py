"""
M2 gate (README): "shuffled-label test gives Sharpe ~= 0". This is the
anti-fooling check for the whole modeling pipeline (features -> dataset ->
LSTM -> walk-forward split -> sign-backtest): if labels are randomly
permuted, X and y are statistically independent by construction, so NO
model should be able to find a tradeable edge on held-out data. If this
test ever fails, that means something in the pipeline leaks the label into
the features (or into the split) -- it does NOT mean "train a better model".

A complementary positive control (`test_model_can_learn_real_signal`) checks
the opposite failure mode: that the pipeline is capable of detecting a real,
strong, causal signal when one genuinely exists. Without this, the shuffled
test could pass for the trivial and useless reason that the model never
learns anything at all.

Runs fully on synthetic data (deterministic seeds) so it's fast (<30s) and
has no dependency on the real curated dataset having been built.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import torch

from src.data.dataset import SequenceDataset
from src.data.features import FEATURE_COLUMNS, compute_features
from src.eval.metrics import sign_backtest
from src.models.encoders.lstm import LSTMBaseline

SEQ_LEN = 15
N_ISINS = 8
N_DAYS = 320


def _synthetic_prices(seed: int, momentum: float = 0.0) -> pl.DataFrame:
    """
    Geometric random walk per ISIN. If momentum != 0, returns are AR(1):
    ret_t = momentum * ret_{t-1} + noise -- a genuine, learnable causal
    signal (used only by the positive-control test).
    """
    rng = np.random.default_rng(seed)
    rows = []
    start_date = dt.date(2015, 1, 1)
    for i in range(N_ISINS):
        isin = f"SYNTH{i:02d}"
        dates = []
        d = start_date
        while len(dates) < N_DAYS:
            if d.weekday() < 5:
                dates.append(d)
            d += dt.timedelta(days=1)

        noise = rng.normal(0, 0.02, size=N_DAYS)
        rets = np.zeros(N_DAYS)
        rets[0] = noise[0]
        for t in range(1, N_DAYS):
            rets[t] = momentum * rets[t - 1] + noise[t]
        price = 100.0 * np.exp(np.cumsum(rets))

        for t in range(N_DAYS):
            close = float(price[t])
            rows.append({
                "isin": isin, "symbol": isin, "date": dates[t],
                "open_adj": close * 0.999, "high_adj": close * 1.005,
                "low_adj": close * 0.995, "close_adj": close,
                "volume": float(rng.integers(1000, 100000)),
            })
    return pl.DataFrame(rows)


def _train_and_backtest(features_df: pl.DataFrame, shuffle_labels: bool, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    ds = SequenceDataset(features_df, seq_len=SEQ_LEN, shuffle_labels=shuffle_labels, shuffle_seed=seed)
    assert len(ds) > 200, f"expected a reasonably sized synthetic dataset, got {len(ds)}"

    # simple chronological 80/20 split by feature_end_date -- this test is
    # about the shuffle mechanism, not walk-forward correctness (that's
    # covered separately in test_walkforward.py)
    dates_sorted = sorted(s.feature_end_date for s in ds.samples)
    cutoff = dates_sorted[int(len(dates_sorted) * 0.8)]
    train_mask = np.array([s.feature_end_date < cutoff for s in ds.samples])
    test_mask = ~train_mask

    train_ds = ds.subset_by_mask(train_mask)
    test_ds = ds.subset_by_mask(test_mask)

    model = LSTMBaseline(n_features=len(FEATURE_COLUMNS), hidden_size=16, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    model.train()
    for _epoch in range(5):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb).numpy())
            actuals.append(yb.numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    return sign_backtest(preds, actuals, cost_bps=10.0)


def test_shuffled_labels_give_near_zero_sharpe():
    prices = _synthetic_prices(seed=1, momentum=0.0)
    features = compute_features(prices)

    result = _train_and_backtest(features, shuffle_labels=True, seed=42)

    # Generous but meaningful bound: with ~300+ test samples, chance Sharpe
    # noise is O(1) in magnitude; a real leak typically produces Sharpe well
    # above 3. See module docstring -- this failing means investigate a
    # leak, not "train harder".
    assert abs(result["net_sharpe"]) < 2.5, (
        f"shuffled-label Sharpe = {result['net_sharpe']:.3f}, expected ~0 -- "
        f"this suggests a lookahead leak somewhere in features/dataset/split"
    )


def test_model_can_learn_real_signal():
    """Positive control: with a strong genuine AR(1) signal, the same
    pipeline (unshuffled) should find real, clearly-positive Sharpe. This
    proves the shuffled test above isn't vacuously passing."""
    prices = _synthetic_prices(seed=2, momentum=0.6)  # strong positive autocorrelation
    features = compute_features(prices)

    result = _train_and_backtest(features, shuffle_labels=False, seed=42)

    assert result["net_sharpe"] > 1.0, (
        f"expected the model to detect a strong synthetic AR(1) signal "
        f"(net_sharpe={result['net_sharpe']:.3f}) -- if this fails, the "
        f"pipeline may be unable to learn ANY signal, which would make the "
        f"shuffled-label test above meaningless"
    )
