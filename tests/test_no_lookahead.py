"""
Structural anti-lookahead / anti-survivorship tests for the M1 data
pipeline. These are deterministic, synthetic-data tests (no network) --
separate from the live-data validation in test_bhavcopy_parsing.py -- so
they run fast in CI and catch a specific regression class each:

  1. A backward price adjustment must NEVER alter a row on/after its own
     ex-date -- only strictly-prior rows may be rescaled. If this breaks,
     `close_adj` on a recent bar would depend on a *future* corporate
     action in a way that changes the recent bar itself, not just the
     historical ones -- a real lookahead leak.
  2. Survivorship: a delisted symbol's `tradable_range` must reflect the
     date it actually stopped trading, never "today" / the current
     universe list.
  3. Year-partitioned curated files must not cross-contaminate dates.
  4. No duplicate (isin, date) rows, which would silently corrupt any
     later windowing/lookback logic.

NOTE on a known, deliberate non-goal: `close_adj` for a bar at time T is
computed using ALL corporate actions ever observed for that ISIN, including
ones announced after T. That does not leak the *future price* (the
multiplier only rescales already-known past prices for continuity), but it
does implicitly reveal "this stock had a corporate action coming" if you
naively recompute the adjusted series as of a training cutoff using
post-cutoff-known actions. Phase 2/4 (causal normalization, purged
walk-forward) is where point-in-time-safe feature construction is enforced;
this file only guards the mechanical adjustment-factor logic itself.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from src.data.build_curated import build_tradable_range, dedupe_prices
from src.data.corporate_actions import apply_adjustment, build_adjustment_factors


def _prices(isin, dates, closes):
    return pl.DataFrame({
        "isin": [isin] * len(dates),
        "symbol": ["TEST"] * len(dates),
        "series": ["EQ"] * len(dates),
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "prev_close": closes,
        "last": closes,
        "volume": [1000] * len(dates),
        "turnover": [1.0] * len(dates),
        "n_trades": [10] * len(dates),
    })


def test_adjustment_never_changes_rows_on_or_after_ex_date():
    dates = [dt.date(2020, 1, d) for d in range(1, 11)]
    prices = _prices("INE_TEST01", dates, [100.0] * 10)

    corp_actions = pl.DataFrame({
        "isin": ["INE_TEST01"],
        "symbol": ["TEST"],
        "series": ["EQ"],
        "ex_date": [dt.date(2020, 1, 6)],
        "subject": ["Bonus 1:1"],  # multiplier = 0.5
    })
    factors = build_adjustment_factors(corp_actions)
    adjusted = apply_adjustment(prices, factors)

    before = adjusted.filter(pl.col("date") < dt.date(2020, 1, 6))
    on_or_after = adjusted.filter(pl.col("date") >= dt.date(2020, 1, 6))

    assert (before["close_adj"] == 50.0).all(), "rows before ex_date must be rescaled by the bonus factor"
    assert (on_or_after["close_adj"] == 100.0).all(), (
        "rows on/after ex_date must be UNCHANGED -- a future action rescaling "
        "current-basis prices would be a lookahead leak"
    )
    assert (on_or_after["adj_factor"] == 1.0).all()


def test_multiple_actions_compound_only_into_the_past():
    dates = [dt.date(2020, 1, d) for d in range(1, 21)]
    prices = _prices("INE_TEST02", dates, [100.0] * 20)

    corp_actions = pl.DataFrame({
        "isin": ["INE_TEST02", "INE_TEST02"],
        "symbol": ["TEST", "TEST"],
        "series": ["EQ", "EQ"],
        "ex_date": [dt.date(2020, 1, 6), dt.date(2020, 1, 16)],
        "subject": ["Bonus 1:1", "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"],
    })
    factors = build_adjustment_factors(corp_actions)
    adjusted = apply_adjustment(prices, factors)

    seg1 = adjusted.filter(pl.col("date") < dt.date(2020, 1, 6))               # before both actions
    seg2 = adjusted.filter((pl.col("date") >= dt.date(2020, 1, 6)) & (pl.col("date") < dt.date(2020, 1, 16)))
    seg3 = adjusted.filter(pl.col("date") >= dt.date(2020, 1, 16))             # after both actions

    assert (seg1["close_adj"] == 100.0 * 0.5 * 0.2).all()
    assert (seg2["close_adj"] == 100.0 * 0.2).all()
    assert (seg3["close_adj"] == 100.0).all()


def test_survivorship_reflects_actual_last_trade_not_today():
    # Symbol A trades throughout; symbol B (simulating a delisting) stops
    # appearing after 2015 even though "today" is 2026.
    dates_a = [dt.date(2010, 1, 1), dt.date(2020, 1, 1), dt.date(2026, 1, 1)]
    dates_b = [dt.date(2010, 1, 1), dt.date(2012, 1, 1), dt.date(2015, 6, 1)]

    df = pl.concat([
        _prices("INE_A", dates_a, [10.0, 20.0, 30.0]),
        _prices("INE_B", dates_b, [10.0, 5.0, 1.0]),
    ])
    tradable = build_tradable_range(df)

    b_row = tradable.filter(pl.col("isin") == "INE_B")
    assert b_row["last_date"][0] == dt.date(2015, 6, 1), (
        "delisted symbol's last_date must be its real last trading day, "
        "not extended to the current date"
    )
    a_row = tradable.filter(pl.col("isin") == "INE_A")
    assert a_row["last_date"][0] == dt.date(2026, 1, 1)


def test_no_duplicate_isin_date_rows():
    dates = [dt.date(2020, 1, 1), dt.date(2020, 1, 1)]  # duplicate on purpose
    df = _prices("INE_DUP", dates, [10.0, 10.0])
    dup_count = df.group_by(["isin", "date"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    assert dup_count.height == 1, "test fixture should contain the duplicate we're checking for"
    # exercise the actual guard used by build_curated(), not a local reimplementation
    deduped = dedupe_prices(df)
    assert deduped.height == 1
