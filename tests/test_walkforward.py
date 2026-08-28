"""
Purge/embargo correctness for the walk-forward splitter, independent of any
model -- pure split-logic tests on a synthetic SequenceDataset.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src.data.dataset import SequenceDataset
from src.data.features import compute_features
from src.eval.walkforward import Fold, assert_no_leakage, split_masks


def _linear_prices(n_days: int = 500, isin: str = "A") -> pl.DataFrame:
    start = dt.date(2015, 1, 1)
    dates, d = [], start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    price = 100.0 + np.arange(n_days) * 0.1
    return pl.DataFrame({
        "isin": [isin] * n_days, "symbol": [isin] * n_days, "date": dates,
        "open_adj": price, "high_adj": price * 1.01, "low_adj": price * 0.99,
        "close_adj": price, "volume": [10000.0] * n_days,
    })


def _dataset(n_days=500, seq_len=10):
    prices = _linear_prices(n_days)
    features = compute_features(prices)
    return SequenceDataset(features, seq_len=seq_len)


def test_train_mask_respects_embargo():
    ds = _dataset()
    fold = Fold(label="test", test_start=dt.date(2016, 6, 1), test_end=dt.date(2016, 12, 31))
    embargo_days = 15

    train_mask, test_mask = split_masks(ds, fold, embargo_days=embargo_days)
    embargo_cutoff = fold.test_start - dt.timedelta(days=embargo_days)

    train_samples = [s for s, m in zip(ds.samples, train_mask) if m]
    assert len(train_samples) > 0
    assert all(s.label_date < embargo_cutoff for s in train_samples)

    # runtime guard, called the way a training script would: on the actual
    # subset it's about to train on. Must not raise on a correctly-built subset.
    train_ds = ds.subset_by_mask(train_mask)
    assert_no_leakage(train_ds, fold, embargo_days=embargo_days)


def test_test_mask_is_strictly_within_fold():
    ds = _dataset()
    fold = Fold(label="test", test_start=dt.date(2016, 6, 1), test_end=dt.date(2016, 12, 31))
    _, test_mask = split_masks(ds, fold, embargo_days=15)

    test_samples = [s for s, m in zip(ds.samples, test_mask) if m]
    assert len(test_samples) > 0
    for s in test_samples:
        assert fold.test_start <= s.feature_end_date
        assert s.label_date <= fold.test_end


def test_no_sample_in_both_train_and_test():
    ds = _dataset()
    fold = Fold(label="test", test_start=dt.date(2016, 6, 1), test_end=dt.date(2016, 12, 31))
    train_mask, test_mask = split_masks(ds, fold, embargo_days=15)
    assert not np.any(train_mask & test_mask)


def test_assert_no_leakage_catches_a_broken_split():
    """assert_no_leakage is an INDEPENDENT guard -- it doesn't call
    split_masks internally -- so this test builds a deliberately-broken
    "training subset" by hand (bypassing split_masks entirely) and confirms
    the guard actually catches it, rather than trivially restating
    split_masks' own definition."""
    ds = _dataset()
    fold = Fold(label="test", test_start=dt.date(2016, 6, 1), test_end=dt.date(2016, 12, 31))

    import dataclasses
    bad_sample = dataclasses.replace(
        ds.samples[0],
        label_date=fold.test_start - dt.timedelta(days=1),  # inside the 15-day embargo window
        feature_end_date=fold.test_start - dt.timedelta(days=2),
    )
    broken_train_ds = ds.subset_by_mask(np.zeros(len(ds.samples), dtype=bool))  # empty, then...
    broken_train_ds.samples.append(bad_sample)  # ...hand-inject the leak, simulating a caller bug

    try:
        assert_no_leakage(broken_train_ds, fold, embargo_days=15)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "assert_no_leakage should have caught a sample whose label bleeds into the embargo window"
