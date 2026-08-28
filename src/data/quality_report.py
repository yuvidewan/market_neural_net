"""
Data-quality gate for the curated Tier-A dataset (README M1 gate: "Data
quality report green"). Runs a fixed set of checks per ISIN and an overall
pass/fail summary. Nothing here modifies the data -- this only reports.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

# Thresholds -- deliberately conservative and named, so the gate criteria are
# auditable rather than a magic if-statement.
MAX_ZERO_VOLUME_FRACTION = 0.10     # >10% zero-volume days for a name is suspicious
MAX_STALE_STREAK_DAYS = 15          # close unchanged for >15 sessions running
MIN_SESSIONS_TO_JUDGE = 20          # names with fewer sessions than this are skipped, not failed


def load_curated_prices(curated_dir: Path) -> pl.DataFrame:
    prices_dir = Path(curated_dir) / "prices"
    parts = sorted(prices_dir.glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(f"No curated price parquet found under {prices_dir}")
    return pl.concat([pl.read_parquet(p) for p in parts])


def _ohlc_violations(df: pl.DataFrame) -> pl.DataFrame:
    have = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    d = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in have]))
    return d.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
    )


def _stale_streaks(df: pl.DataFrame) -> pl.DataFrame:
    """Max consecutive-unchanged-close run length per isin."""
    d = df.sort(["isin", "date"]).with_columns(
        (pl.col("close") != pl.col("close").shift(1).over("isin")).fill_null(True).alias("changed")
    )
    d = d.with_columns(pl.col("changed").cum_sum().over("isin").alias("run_id"))
    streaks = d.group_by(["isin", "run_id"]).agg(pl.len().alias("streak_len"))
    return streaks.group_by("isin").agg(pl.col("streak_len").max().alias("max_stale_streak"))


def generate_quality_report(curated_dir: Path, out_dir: Path | None = None) -> dict:
    df = load_curated_prices(curated_dir)
    out_dir = Path(out_dir) if out_dir else Path(curated_dir) / "quality_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    trading_calendar = df.select("date").unique().sort("date")
    n_trading_days = trading_calendar.height

    per_isin = (
        df.group_by("isin")
        .agg([
            pl.col("symbol").last().alias("symbol"),
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.col("date").n_unique().alias("n_sessions"),
            (pl.col("volume") == 0).sum().alias("n_zero_volume"),
            (pl.col("close") <= 0).sum().alias("n_nonpositive_close"),
        ])
    )
    per_isin = per_isin.with_columns(
        (pl.col("n_zero_volume") / pl.col("n_sessions")).alias("zero_volume_fraction")
    )

    # expected sessions = trading days in [first_date, last_date] that are on
    # the master calendar (any day ANY equity traded) -- this is the
    # survivorship-safe reference: it's built from what actually traded, not
    # from an assumed calendar.
    calendar_dates = trading_calendar["date"]
    expected = per_isin.select(["isin", "first_date", "last_date"]).with_columns(
        pl.struct(["first_date", "last_date"]).map_elements(
            lambda s: calendar_dates.filter(
                (calendar_dates >= s["first_date"]) & (calendar_dates <= s["last_date"])
            ).len(),
            return_dtype=pl.Int64,
        ).alias("expected_sessions")
    )
    per_isin = per_isin.join(expected.select(["isin", "expected_sessions"]), on="isin")
    per_isin = per_isin.with_columns(
        (pl.col("expected_sessions") - pl.col("n_sessions")).alias("missing_sessions")
    )

    ohlc_bad = _ohlc_violations(df)
    ohlc_bad_by_isin = ohlc_bad.group_by("isin").agg(pl.len().alias("n_ohlc_violations"))
    per_isin = per_isin.join(ohlc_bad_by_isin, on="isin", how="left").with_columns(
        pl.col("n_ohlc_violations").fill_null(0)
    )

    stale = _stale_streaks(df)
    per_isin = per_isin.join(stale, on="isin", how="left").with_columns(
        pl.col("max_stale_streak").fill_null(0)
    )

    judged = per_isin.filter(pl.col("n_sessions") >= MIN_SESSIONS_TO_JUDGE)
    failing = judged.filter(
        (pl.col("zero_volume_fraction") > MAX_ZERO_VOLUME_FRACTION)
        | (pl.col("n_nonpositive_close") > 0)
        | (pl.col("n_ohlc_violations") > 0)
        | (pl.col("max_stale_streak") > MAX_STALE_STREAK_DAYS)
    )

    per_isin.write_parquet(out_dir / "per_isin_report.parquet")
    failing.write_csv(out_dir / "failing_isins.csv")

    summary = {
        "n_trading_days_in_calendar": n_trading_days,
        "calendar_start": str(trading_calendar["date"].min()) if n_trading_days else None,
        "calendar_end": str(trading_calendar["date"].max()) if n_trading_days else None,
        "n_isins_total": per_isin.height,
        "n_isins_judged": judged.height,
        "n_isins_failing": failing.height,
        "n_ohlc_violations_total": int(ohlc_bad.height),
        "pct_isins_failing": round(100 * failing.height / max(judged.height, 1), 3),
        "gate": "GREEN" if (ohlc_bad.height == 0 and failing.height / max(judged.height, 1) < 0.02) else "RED",
    }
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items())
    )
    return summary
