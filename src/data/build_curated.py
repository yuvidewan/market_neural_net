"""
Assemble the curated Tier-A daily dataset from interim parquet:
  1. Load all interim/bhavcopy/{year}.parquet files.
  2. Restrict to the standard equity series (EQ) -- BE/BZ/etc kept in interim
     but excluded from the default curated set (documented, not hidden).
  3. Fetch NSE corporate actions and compute backward adjustment factors.
  4. Write year-partitioned curated parquet with both raw and adjusted OHLC.
  5. Derive `tradable_range` (first/last session per ISIN) directly from
     observed trading days -- this is what makes survivorship bias a
     non-issue here: a delisted stock's history simply stops appearing,
     it was never filtered out based on today's membership.

KNOWN DATA-SOURCE LIMITATION, handled explicitly rather than by silently
dropping data: NSE's bhavcopy did not include an ISIN column until roughly
2011 (confirmed: absent through Jul-2010, present by Jul-2011 -- the exact
month isn't pinned down further since it doesn't matter for what follows).
Dropping every pre-2011 row for lacking an ISIN would throw away ~13 years
of the "full history" this project asked for. Instead, `security_id` is
backfilled via a symbol->ISIN lookup built from the (post-2011) rows that do
carry one, and only falls back to a synthetic `SYM:<symbol>` id for names
that never appear in the ISIN era at all (i.e. delisted before ~2011). That
fallback case is flagged in `tradable_range` via `isin_is_symbol_fallback`
so it's never silently treated as equal-quality to a real ISIN match --
e.g. it can't be joined against the corporate-actions feed (which is
ISIN-keyed), so those names get no split/bonus adjustment.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data.corporate_actions import (
    apply_adjustment,
    build_adjustment_factors,
    fetch_corporate_actions,
)

DEFAULT_SERIES = ["EQ"]


def load_interim(interim_dir: Path) -> pl.DataFrame:
    interim_dir = Path(interim_dir)
    files = sorted(interim_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No interim parquet found in {interim_dir}")
    return pl.concat([pl.read_parquet(f) for f in files])


def dedupe_prices(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (isin, date). Guards against a rare duplicate bhavcopy
    listing (or a re-run merge bug) silently corrupting downstream windowing."""
    return df.unique(subset=["isin", "date"], keep="first")


def backfill_isin_via_symbol(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fill missing `isin` (pre-~2011 bhavcopy rows) using a symbol->ISIN
    lookup built from rows that do carry a real ISIN. Rows for a symbol that
    never appears in the ISIN era get a synthetic `SYM:<symbol>` id instead,
    flagged via `isin_is_symbol_fallback`. See module docstring for why.
    """
    known = (
        df.filter(pl.col("isin").is_not_null() & (pl.col("isin") != ""))
        .group_by("symbol")
        .agg(pl.col("isin").mode().first().alias("_known_isin"))
    )
    out = df.join(known, on="symbol", how="left")
    out = out.with_columns(
        pl.when(pl.col("isin").is_not_null() & (pl.col("isin") != ""))
        .then(pl.col("isin"))
        .when(pl.col("_known_isin").is_not_null())
        .then(pl.col("_known_isin"))
        .otherwise("SYM:" + pl.col("symbol"))
        .alias("isin")
    ).with_columns(
        (~pl.col("isin").str.starts_with("SYM:") & pl.col("isin").is_not_null()).not_().alias("isin_is_symbol_fallback")
    ).drop("_known_isin")
    return out


def build_tradable_range(df: pl.DataFrame) -> pl.DataFrame:
    agg_cols = [
        pl.col("symbol").last().alias("last_symbol"),
        pl.col("symbol").unique().alias("symbols_seen"),
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date"),
        pl.col("date").n_unique().alias("n_sessions"),
    ]
    if "isin_is_symbol_fallback" in df.columns:
        agg_cols.append(pl.col("isin_is_symbol_fallback").any())
    return df.group_by("isin").agg(agg_cols).sort("first_date")


def build_curated(
    interim_dir: Path,
    curated_dir: Path,
    series: list[str] | None = None,
    fetch_actions: bool = True,
    corp_actions_start_year: int = 1996,
) -> dict:
    series = series or DEFAULT_SERIES
    curated_dir = Path(curated_dir)
    prices_dir = curated_dir / "prices"
    universe_dir = curated_dir / "universe"
    ca_dir = curated_dir / "corporate_actions"
    for d in (prices_dir, universe_dir, ca_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_interim(interim_dir)
    eq = raw.filter(pl.col("series").is_in(series)).drop_nulls(["close", "symbol"])
    eq = eq.filter(pl.col("symbol") != "")
    eq = backfill_isin_via_symbol(eq)
    eq = dedupe_prices(eq)

    tradable_range = build_tradable_range(eq)
    tradable_range.write_parquet(universe_dir / "tradable_range.parquet")

    end_year = eq["date"].max().year
    if fetch_actions:
        print(f"Fetching corporate actions {corp_actions_start_year}-{end_year} ...")
        corp_actions = fetch_corporate_actions(corp_actions_start_year, end_year)
        corp_actions.write_parquet(ca_dir / "raw_actions.parquet")
        factors = build_adjustment_factors(corp_actions, out_dir=ca_dir)
        factors.write_parquet(ca_dir / "adjustment_factors.parquet")
    else:
        factors = pl.DataFrame(schema={"isin": pl.Utf8, "ex_date": pl.Date, "cum_factor": pl.Float64})

    adjusted = apply_adjustment(eq, factors)

    years = sorted(adjusted["date"].dt.year().unique().to_list())
    for year in years:
        year_df = adjusted.filter(pl.col("date").dt.year() == year)
        out = prices_dir / f"year={year}"
        out.mkdir(parents=True, exist_ok=True)
        year_df.write_parquet(out / "part.parquet")

    return {
        "rows": adjusted.height,
        "isins": tradable_range.height,
        "years": years,
        "n_adjustment_events": factors.height,
    }
