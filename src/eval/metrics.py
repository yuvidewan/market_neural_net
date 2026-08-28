"""
Backtest metrics for the M2 baseline. Deliberately small right now -- the
full metric set from README §6 (Sortino, Calmar, DSR, PBO, capacity, ...)
belongs to Phase 4/5 once there's a real trading policy to evaluate; this
module covers just what the shuffled-label gate and the M2 baseline report
need: Sharpe, hit rate, and a simple sign-based backtest with a flat cost.
"""
from __future__ import annotations

import numpy as np


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
