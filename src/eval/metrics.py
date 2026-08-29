"""
Backtest metrics for the M2 baseline. Deliberately small right now -- the
full metric set from README §6 (Sortino, Calmar, DSR, PBO, capacity, ...)
belongs to Phase 4/5 once there's a real trading policy to evaluate; this
module covers just what the shuffled-label gate and the M2 baseline report
need: Sharpe, hit rate, and a simple sign-based backtest with a flat cost.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1))


def hit_rate(pred: np.ndarray, actual: np.ndarray, deadzone: float = 0.0) -> float:
    pred, actual = np.asarray(pred), np.asarray(actual)
    active = np.abs(pred) > deadzone
    if active.sum() == 0:
        return float("nan")
    return float((np.sign(pred[active]) == np.sign(actual[active])).mean())


def sign_backtest(
    pred: np.ndarray,
    actual_ret: np.ndarray,
    cost_bps: float = 10.0,
    deadzone: float = 0.0,
) -> dict:
    """
    Simplest possible trading policy, used only to sanity-check that a model
    has *some* real signal before anything fancier (Phase 5's RL agent) is
    worth building: position_t in {-1, 0, +1} = sign(pred_t) if
    |pred_t| > deadzone else 0. Strategy return on day t+1 is
    position_t * actual_ret_t+1, minus a flat round-trip cost on any change
    in position. This is NOT the Indian cost model (README §4.3) -- that's
    still Phase 4.3/5 work; `cost_bps` here is a placeholder default (10bps
    round-trip) so a signal can't look good purely because costs were
    ignored.
    """
    pred, actual_ret = np.asarray(pred, dtype=np.float64), np.asarray(actual_ret, dtype=np.float64)
    # NOTE: turnover is computed as a naive sample-to-sample diff. If `pred`
    # spans multiple symbols concatenated together (as it does when this is
    # called on a walk-forward test set built isin-by-isin), the handful of
    # symbol-boundary transitions get treated as one continuous book, which
    # is not quite right -- a minor inaccuracy in this placeholder metric,
    # not in the underlying signal. Immaterial at the boundary counts M2/M3
    # produce (~40 boundaries in thousands of samples); Phase 4/5's real
    # per-symbol position tracking replaces this properly.
    position = np.where(np.abs(pred) > deadzone, np.sign(pred), 0.0)
    prev_position = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - prev_position)
    gross_pnl = position * actual_ret
    cost = turnover * (cost_bps / 1e4)
    net_pnl = gross_pnl - cost

    return {
        "n": len(pred),
        "gross_sharpe": sharpe_ratio(gross_pnl),
        "net_sharpe": sharpe_ratio(net_pnl),
        "hit_rate": hit_rate(pred, actual_ret, deadzone),
        "mean_abs_position": float(np.abs(position).mean()),
        "avg_turnover_per_day": float(turnover.mean()),
        "net_pnl": net_pnl,
    }


def rank_ic_by_date(dates: np.ndarray, pred: np.ndarray, actual: np.ndarray, min_names: int = 3) -> dict:
    """
    Cross-sectional rank Information Coefficient, computed the way the
    industry actually uses it (README §6/§Phase3): for each date, Spearman
    rank-correlate the model's predictions against realized returns ACROSS
    THE NAMES TRADED THAT DAY, not pooled across all (date, symbol) pairs.
    A signal's job is to rank stocks against each other on a given day; a
    pooled correlation would conflate that with day-to-day market moves.

    Returns {date: ic} for every date with >= min_names observations
    (Spearman on fewer than 3 points is not meaningful and is skipped, not
    silently coerced to 0 or NaN-propagated into the mean).
    """
    dates = np.asarray(dates)
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    ic_by_date = {}
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < min_names:
            continue
        p, a = pred[mask], actual[mask]
        if np.std(p) == 0 or np.std(a) == 0:
            continue
        ic, _ = spearmanr(p, a)
        if not np.isnan(ic):
            ic_by_date[d] = float(ic)
    return ic_by_date


def rank_ic_summary(dates: np.ndarray, pred: np.ndarray, actual: np.ndarray, min_names: int = 3) -> dict:
    """
    mean_ic: average daily cross-sectional IC -- the M3 gate threshold (>0.02)
      is checked against this.
    ic_ir: mean_ic / std(daily ic) -- "IC-IR" per README §6, a stability
      measure (a mean IC propped up by a couple of huge-IC days is a much
      weaker result than the same mean achieved consistently).
    pct_positive_days: fraction of days with IC > 0 -- another stability lens,
      independent of the mean/std ratio.
    """
    ic_by_date = rank_ic_by_date(dates, pred, actual, min_names=min_names)
    if not ic_by_date:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "pct_positive_days": float("nan"), "n_days": 0}
    ics = np.array(list(ic_by_date.values()))
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    return {
        "mean_ic": mean_ic,
        "ic_ir": mean_ic / std_ic if std_ic > 0 else 0.0,
        "pct_positive_days": float((ics > 0).mean()),
        "n_days": len(ics),
    }
