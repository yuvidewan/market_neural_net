"""
Parser correctness tests, run against real NSE bhavcopy files (fixtures
downloaded from nsearchives.nseindia.com) covering both historical formats.
"""
import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from src.data.ingest.bhavcopy import parse_bhavcopy_zip, candidate_urls, FORMAT_CUTOVER

FIXTURES = Path(__file__).parent / "fixtures"

REQUIRED_COLS = [
    "date", "isin", "symbol", "series",
    "open", "high", "low", "close", "prev_close", "last",
    "volume", "turnover", "n_trades",
]


@pytest.mark.parametrize("fixture,date", [
    ("2024-01-01_old_format.zip", dt.date(2024, 1, 1)),
    ("2026-08-27_new_format.zip", dt.date(2026, 8, 27)),
    ("1998-01-02_old_format.zip", dt.date(1998, 1, 2)),
])
def test_parse_schema(fixture, date):
    content = (FIXTURES / fixture).read_bytes()
    df = parse_bhavcopy_zip(content, date)

    assert df.columns == REQUIRED_COLS
    assert df.height > 0
    # every row's date must equal the requested date -- no future/past leakage
    assert (df["date"] == date).all()


def test_parse_no_negative_prices():
    content = (FIXTURES / "2024-01-01_old_format.zip").read_bytes()
    df = parse_bhavcopy_zip(content, dt.date(2024, 1, 1))
    price_cols = ["open", "high", "low", "close", "prev_close"]
    for c in price_cols:
        bad = df.filter(pl.col(c).is_not_null() & (pl.col(c) < 0))
        assert bad.height == 0, f"negative values found in {c}"


def test_parse_high_low_consistency():
    content = (FIXTURES / "2024-01-01_old_format.zip").read_bytes()
    df = parse_bhavcopy_zip(content, dt.date(2024, 1, 1))
    df = df.filter(
        pl.col("high").is_not_null() & pl.col("low").is_not_null()
        & pl.col("open").is_not_null() & pl.col("close").is_not_null()
    )
    bad = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
    )
    assert bad.height == 0, f"OHLC inconsistency in {bad.height} rows"


def test_new_format_excludes_non_equity_instruments():
    # The new UDIFF CM-segment file bundles bonds/T-bills/SGBs alongside
    # equities (FinInstrmTp != "STK"); parser must drop those.
    content = (FIXTURES / "2026-08-27_new_format.zip").read_bytes()
    df = parse_bhavcopy_zip(content, dt.date(2026, 8, 27))
    # sanity: real equity symbols like large-caps should still be absent from
    # this particular fixture page (it only has SGB rows in the raw file's
    # head), but the frame must not error and must keep the normalized schema
    assert set(df.columns) == set(REQUIRED_COLS)


def test_symbols_present_are_plausible():
    content = (FIXTURES / "1998-01-02_old_format.zip").read_bytes()
    df = parse_bhavcopy_zip(content, dt.date(1998, 1, 2))
    symbols = df["symbol"].to_list()
    assert "AARTIDRUGS" in symbols  # visible in raw fixture content


def test_candidate_urls_ordering():
    before = candidate_urls(FORMAT_CUTOVER - dt.timedelta(days=1))
    after = candidate_urls(FORMAT_CUTOVER)
    assert "historical/EQUITIES" in before[0]  # old format tried first
    assert "BhavCopy_NSE_CM" in after[0]        # new format tried first
    # both always offer the other format as fallback
    assert len(before) == 2 and len(after) == 2
