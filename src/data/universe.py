"""
Index universe construction.

KNOWN LIMITATION (documented, not silently papered over): NSE does not
publish free historical index-membership snapshots. The constituent CSVs
fetched here (`fetch_index_constituents`) are TODAY's membership only.
Using today's NIFTY 500 list as if it applied to 2010 would introduce
survivorship bias in exactly the way Phase 1 of the README says to avoid.

What this module gives you instead, which is genuinely point-in-time-safe:
  - `data/curated/prices/` (built by build_curated.py) is derived from the
    exchange's actual per-day bhavcopy files, so a stock that delisted in
    2015 still has its full trading history in the dataset up to its last
    session -- survivorship bias in the *price* data is avoided structurally,
    for free, without needing historical index lists at all.
  - `index_membership` below is written with an explicit `as_of_date` so
    every consumer is forced to see that it's a snapshot, not a time series.

Backfilling true historical constituent changes (index inclusion/exclusion
circulars) is future work -- tracked as a TODO rather than faked.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import requests

from src.data.ingest.bhavcopy import USER_AGENT

INDEX_CSV_URLS = {
    "NIFTY50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY200": "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
}


def fetch_index_constituents(index: str, timeout: float = 15.0) -> pl.DataFrame:
    if index not in INDEX_CSV_URLS:
        raise ValueError(f"Unknown index {index!r}; known: {list(INDEX_CSV_URLS)}")
    r = requests.get(INDEX_CSV_URLS[index], headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    from io import BytesIO
    df = pl.read_csv(BytesIO(r.content))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename({
        "Company Name": "company_name",
        "Industry": "industry",
        "Symbol": "symbol",
        "Series": "series",
        "ISIN Code": "isin",
    })
    df = df.with_columns([
        pl.lit(index).alias("index"),
        pl.lit(dt.date.today()).alias("as_of_date"),
    ])
    return df.select(["as_of_date", "index", "symbol", "isin", "company_name", "industry", "series"])


def build_universe(out_dir: Path, indices: list[str] | None = None) -> pl.DataFrame:
    indices = indices or list(INDEX_CSV_URLS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [fetch_index_constituents(ix) for ix in indices]
    combined = pl.concat(frames)
    combined.write_parquet(out_dir / "index_membership_current.parquet")
    return combined


def liquid_universe_from_prices(
    prices: pl.DataFrame,
    as_of_date: dt.date,
    lookback_days: int = 60,
    min_adv_inr: float = 5e7,  # ADV > INR 5 crore
) -> pl.DataFrame:
    """
    Liquidity-based universe fallback that needs no external index list at
    all: any ISIN whose trailing `lookback_days` average daily turnover
    (already in the bhavcopy data as `turnover`, in INR) exceeds
    `min_adv_inr`, as of `as_of_date`. This *is* point-in-time correct,
    since it only looks at data on or before as_of_date -- the honest
    complement to the today-only index CSVs above.
    """
    window_start = as_of_date - dt.timedelta(days=int(lookback_days * 1.6))  # pad for weekends/holidays
    window = prices.filter(
        (pl.col("date") <= as_of_date) & (pl.col("date") > window_start) & (pl.col("series") == "EQ")
    )
    adv = (
        window.group_by("isin")
        .agg([
            pl.col("turnover").mean().alias("adv_inr"),
            pl.col("symbol").last().alias("symbol"),
            pl.col("date").count().alias("n_sessions"),
        ])
        .filter((pl.col("adv_inr") > min_adv_inr) & (pl.col("n_sessions") >= lookback_days // 2))
    )
    return adv.with_columns(pl.lit(as_of_date).alias("as_of_date"))
