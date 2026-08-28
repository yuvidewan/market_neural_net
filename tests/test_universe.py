"""
Logic tests for the universe module. `fetch_index_constituents` itself hits
a live NSE endpoint and is exercised via scripts/build_universe.py, not here
-- unit tests stay deterministic and network-free.
"""
import datetime as dt

import polars as pl

from src.data.universe import liquid_universe_from_prices


def _price_row(isin, symbol, date, turnover, series="EQ"):
    return {
        "isin": isin, "symbol": symbol, "series": series, "date": date,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "prev_close": 100.0, "last": 100.0, "volume": 1000, "turnover": turnover, "n_trades": 10,
    }


def test_liquid_universe_filters_by_adv_and_series():
    as_of = dt.date(2020, 6, 30)
    rows = []
    # ISIN A: liquid, 60 sessions, high turnover -> should pass
    for i in range(60):
        rows.append(_price_row("INE_A", "AAA", as_of - dt.timedelta(days=i), turnover=1e8))
    # ISIN B: illiquid, low turnover -> should fail
    for i in range(60):
        rows.append(_price_row("INE_B", "BBB", as_of - dt.timedelta(days=i), turnover=1e5))
    # ISIN C: liquid but wrong series (BE, not EQ) -> should be excluded
    for i in range(60):
        rows.append(_price_row("INE_C", "CCC", as_of - dt.timedelta(days=i), turnover=1e8, series="BE"))
    # ISIN D: liquid turnover but only a handful of sessions -> should fail (too sparse)
    for i in range(5):
        rows.append(_price_row("INE_D", "DDD", as_of - dt.timedelta(days=i), turnover=1e8))

    prices = pl.DataFrame(rows)
    universe = liquid_universe_from_prices(prices, as_of_date=as_of, lookback_days=60, min_adv_inr=5e7)

    isins = set(universe["isin"].to_list())
    assert isins == {"INE_A"}


def test_liquid_universe_is_point_in_time_safe():
    # A row dated AFTER as_of_date must never influence the result.
    as_of = dt.date(2020, 6, 30)
    rows = [_price_row("INE_A", "AAA", as_of - dt.timedelta(days=i), turnover=1e8) for i in range(60)]
    rows.append(_price_row("INE_A", "AAA", as_of + dt.timedelta(days=5), turnover=1e12))  # future, huge turnover
    prices = pl.DataFrame(rows)

    universe = liquid_universe_from_prices(prices, as_of_date=as_of, lookback_days=60, min_adv_inr=5e7)
    a_row = universe.filter(pl.col("isin") == "INE_A")
    assert a_row["adv_inr"][0] < 1e9, "future-dated row leaked into a point-in-time ADV computation"
