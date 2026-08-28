"""
Causal feature engineering (README Phase 2): raw market state only, no
handcrafted indicators. Every feature at row t uses only data available
through end-of-day t; the one column that legitimately looks forward
(`fwd_ret_1`, the training target) is clearly named and documented as such
so it's never mistaken for a feature.

All features are computed on `close_adj`/`open_adj`/`high_adj`/`low_adj`
(the corporate-action-adjusted series from build_curated.py) so a stock
split doesn't show up as a fake -80% return.
"""
from __future__ import annotations

import polars as pl

FEATURE_COLUMNS = [
    "ret_1", "ret_o", "ret_h", "ret_l",
    "ret_1_volscaled",
    "vol_z",
    "dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos",
]

TARGET_COLUMN = "fwd_ret_1"

# Rolling windows, all trailing-inclusive of the current row (causal: a
# feature "as of end of day t" may use day t's own OHLCV, which has already
# happened by the time it's computed -- it just may never use day t+1).
VOL_WINDOW = 20
VOLUME_BASELINE_WINDOW = 252
MIN_HISTORY_FOR_FEATURES = 60  # rows with less trailing history than this get NaN features


def compute_features(prices: pl.DataFrame) -> pl.DataFrame:
    """
    `prices` must have columns: isin, date, open_adj, high_adj, low_adj,
    close_adj, volume (one row per isin/date, already deduped, sorted or not).
    Returns the same rows with FEATURE_COLUMNS + TARGET_COLUMN added; rows
    without enough trailing history, or without a next-session label, have
    the corresponding columns as null rather than being silently dropped --
    callers decide how to handle that (the Dataset class drops them).
    """
    df = prices.sort(["isin", "date"])

    prev_close = pl.col("close_adj").shift(1).over("isin")
    df = df.with_columns([
        (pl.col("close_adj").log() - prev_close.log()).alias("ret_1"),
        (pl.col("open_adj").log() - prev_close.log()).alias("ret_o"),
        (pl.col("high_adj").log() - prev_close.log()).alias("ret_h"),
        (pl.col("low_adj").log() - prev_close.log()).alias("ret_l"),
        pl.col("volume").add(1).log().alias("log_vol"),
    ])

    df = df.with_columns([
        pl.col("ret_1").rolling_std(window_size=VOL_WINDOW, min_samples=VOL_WINDOW).over("isin").alias("realized_vol_20"),
        pl.col("log_vol").rolling_median(window_size=VOLUME_BASELINE_WINDOW, min_samples=VOLUME_BASELINE_WINDOW // 4).over("isin").alias("log_vol_baseline"),
        pl.col("log_vol").rolling_std(window_size=VOLUME_BASELINE_WINDOW, min_samples=VOLUME_BASELINE_WINDOW // 4).over("isin").alias("log_vol_std"),
    ])

    df = df.with_columns([
        (pl.col("ret_1") / (pl.col("realized_vol_20") + 1e-8)).alias("ret_1_volscaled"),
        ((pl.col("log_vol") - pl.col("log_vol_baseline")) / (pl.col("log_vol_std") + 1e-8)).alias("vol_z"),
    ])

    # calendar structure, cyclical encoding (no raw integer -- avoids implying
    # a false linear order across e.g. December -> January)
    dow = pl.col("date").dt.weekday().cast(pl.Float64)  # 1=Mon .. 7=Sun (NSE never trades 6/7)
    dom = pl.col("date").dt.day().cast(pl.Float64)
    month = pl.col("date").dt.month().cast(pl.Float64)
    two_pi = 2 * 3.141592653589793
    df = df.with_columns([
        (dow * two_pi / 7).sin().alias("dow_sin"), (dow * two_pi / 7).cos().alias("dow_cos"),
        (dom * two_pi / 31).sin().alias("dom_sin"), (dom * two_pi / 31).cos().alias("dom_cos"),
        (month * two_pi / 12).sin().alias("month_sin"), (month * two_pi / 12).cos().alias("month_cos"),
    ])

    # Target: next session's return for this isin. This is the ONLY column
    # that uses t+1 data, by design -- it is a label, never a feature.
    next_close = pl.col("close_adj").shift(-1).over("isin")
    df = df.with_columns(
        (next_close.log() - pl.col("close_adj").log()).alias(TARGET_COLUMN)
    )

    # mark rows with insufficient trailing history as unusable rather than
    # feeding the model a feature vector that's mostly padding/bias
    row_idx_in_isin = pl.int_range(pl.len()).over("isin")
    df = df.with_columns(
        (row_idx_in_isin >= MIN_HISTORY_FOR_FEATURES).alias("has_sufficient_history")
    )

    return df
