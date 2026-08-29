"""
Tests for PanelSequenceDataset (README Phase 3 v2's cross-sectional dataset).
"""
import datetime as dt

import numpy as np
import polars as pl

from src.data.panel_dataset import PanelSequenceDataset


def _rows(isin, dates, closes, has_hist=True):
    n = len(dates)
    out = []
    for i, (d, c) in enumerate(zip(dates, closes)):
        out.append({
            "isin": isin, "date": d,
            "ret_1": 0.01, "ret_o": 0.0, "ret_h": 0.0, "ret_l": 0.0,
            "ret_1_volscaled": 0.0, "vol_z": 0.0,
            "dow_sin": 0.0, "dow_cos": 0.0, "dom_sin": 0.0, "dom_cos": 0.0,
            "month_sin": 0.0, "month_cos": 0.0,
            "fwd_ret_1": 0.02 if i < n - 1 else None,  # last row has no next-day label
            "has_sufficient_history": has_hist and i >= 3,  # pretend min-history=3 for this test
        })
    return out


def _dates(start, n):
    return [start + dt.timedelta(days=i) for i in range(n)]


def test_panel_excludes_insufficient_history():
    dates_a = _dates(dt.date(2020, 1, 1), 10)
    dates_b = _dates(dt.date(2020, 1, 1), 10)
    rows = _rows("A", dates_a, range(10)) + _rows("B", dates_b, range(10))
    df = pl.DataFrame(rows)

    ds = PanelSequenceDataset(df, universe=["A", "B"], seq_len=4, min_symbols_per_date=1)
    # first valid anchor index per isin is row 3 (0-indexed, has_sufficient_history True from i>=3)
    # and seq_len=4 needs idx>=3 anyway -- so the earliest panel date is dates[3]
    assert ds.samples[0].feature_end_date == dates_a[3]


def test_panel_respects_min_symbols_per_date():
    dates_a = _dates(dt.date(2020, 1, 1), 10)
    dates_b = _dates(dt.date(2020, 1, 5), 10)  # starts later -> fewer overlapping valid dates
    rows = _rows("A", dates_a, range(10)) + _rows("B", dates_b, range(10))
    df = pl.DataFrame(rows)

    ds_strict = PanelSequenceDataset(df, universe=["A", "B"], seq_len=4, min_symbols_per_date=2)
    ds_loose = PanelSequenceDataset(df, universe=["A", "B"], seq_len=4, min_symbols_per_date=1)
    assert len(ds_strict) <= len(ds_loose)
    for s in ds_strict.samples:
        assert len(s.entries) >= 2


def test_panel_getitem_shapes_and_no_lookahead():
    dates_a = _dates(dt.date(2020, 1, 1), 10)
    dates_b = _dates(dt.date(2020, 1, 1), 10)
    rows = _rows("A", dates_a, range(10)) + _rows("B", dates_b, range(10))
    df = pl.DataFrame(rows)
    # give ret_1 a value that increases over time so we can detect lookahead
    # (polars Date's physical repr is already days-since-epoch as an int -- no unit conversion needed)
    df = df.with_columns(pl.col("date").cast(pl.Int64).alias("ret_1"))

    ds = PanelSequenceDataset(df, universe=["A", "B"], seq_len=4, min_symbols_per_date=1)
    idx = 0
    X, y = ds[idx]
    anchor_date = ds.samples[idx].feature_end_date
    assert X.shape == (len(ds.samples[idx].entries), 4, 12)  # n_symbols_that_date, seq_len, n_features
    assert y.shape == (len(ds.samples[idx].entries),)

    # ret_1 is column 0; no value in the window may exceed the anchor date's own encoded value
    anchor_val = (anchor_date - dt.date(1970, 1, 1)).days
    assert X[:, :, 0].max().item() <= anchor_val + 1e-6, "window contains data past the anchor date"


def test_subset_by_mask():
    dates_a = _dates(dt.date(2020, 1, 1), 10)
    df = pl.DataFrame(_rows("A", dates_a, range(10)))
    ds = PanelSequenceDataset(df, universe=["A"], seq_len=4, min_symbols_per_date=1)
    mask = np.array([i % 2 == 0 for i in range(len(ds))])
    sub = ds.subset_by_mask(mask)
    assert len(sub) == int(mask.sum())
    assert sub.samples == [s for s, m in zip(ds.samples, mask) if m]
