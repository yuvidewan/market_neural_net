import datetime as dt

import numpy as np

from src.eval.metrics import rank_ic_by_date, rank_ic_summary


def test_perfect_rank_correlation_gives_ic_one():
    dates = np.array([dt.date(2020, 1, 1)] * 5)
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # same order -> perfect rank agreement
    ic_by_date = rank_ic_by_date(dates, pred, actual)
    assert len(ic_by_date) == 1
    assert np.isclose(list(ic_by_date.values())[0], 1.0)


def test_perfect_inverse_rank_correlation_gives_ic_minus_one():
    dates = np.array([dt.date(2020, 1, 1)] * 5)
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    ic_by_date = rank_ic_by_date(dates, pred, actual)
    assert np.isclose(list(ic_by_date.values())[0], -1.0)


def test_days_below_min_names_are_skipped_not_zeroed():
    dates = np.array([dt.date(2020, 1, 1)] * 2 + [dt.date(2020, 1, 2)] * 5)
    pred = np.array([1.0, 2.0] + [1.0, 2.0, 3.0, 4.0, 5.0])
    actual = np.array([1.0, 2.0] + [10.0, 20.0, 30.0, 40.0, 50.0])
    ic_by_date = rank_ic_by_date(dates, pred, actual, min_names=3)
    assert dt.date(2020, 1, 1) not in ic_by_date, "a 2-name day should be skipped at min_names=3"
    assert dt.date(2020, 1, 2) in ic_by_date


def test_cross_sectional_grouping_not_pooled():
    """The key property this metric exists for: a signal that ranks
    correctly WITHIN each day, even if the days have wildly different
    overall return levels, must score IC=1 -- a naive pooled correlation
    across days would be confounded by the day-level shift and score much
    lower."""
    dates = np.array([dt.date(2020, 1, 1)] * 3 + [dt.date(2020, 1, 2)] * 3)
    pred = np.array([1.0, 2.0, 3.0] * 2)  # same within-day ranking both days
    # day 1 actual returns are all strongly negative, day 2 all strongly positive,
    # but the WITHIN-day ranking still perfectly matches pred both times
    actual = np.array([-30.0, -20.0, -10.0, 10.0, 20.0, 30.0])

    summary = rank_ic_summary(dates, pred, actual)
    assert np.isclose(summary["mean_ic"], 1.0)
    assert summary["n_days"] == 2
    assert summary["pct_positive_days"] == 1.0


def test_ic_ir_reflects_stability():
    # two days both with IC=1.0 -> zero variance -> ic_ir falls back to 0.0
    # per the guard (std=0), which is the documented, deliberate behavior
    # for a degenerate case, not an error
    dates = np.array([dt.date(2020, 1, 1)] * 3 + [dt.date(2020, 1, 2)] * 3)
    pred = np.array([1.0, 2.0, 3.0] * 2)
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    summary = rank_ic_summary(dates, pred, actual)
    assert summary["ic_ir"] == 0.0


def test_zero_variance_predictions_are_skipped():
    dates = np.array([dt.date(2020, 1, 1)] * 4)
    pred = np.array([1.0, 1.0, 1.0, 1.0])  # constant prediction -> undefined rank correlation
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    ic_by_date = rank_ic_by_date(dates, pred, actual)
    assert len(ic_by_date) == 0
