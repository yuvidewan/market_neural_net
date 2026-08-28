"""
Corporate actions: fetch NSE's own corporate-action feed and turn Bonus /
Face-Value-Split events into per-ISIN backward price-adjustment factors.

Scope, stated explicitly rather than silently assumed:
  - Handled mechanically: Bonus issues ("Bonus 1:1"), face-value splits and
    consolidations ("Face Value Split ... From Rs 10 To Rs 2").
  - NOT auto-adjusted: dividends (ex-dividend drop is a total-return concept,
    not a share-count mechanical ratio -- conflating the two is a common
    source of subtly wrong adjusted series), rights issues (ratio *and*
    subscription price both matter), and scheme-of-arrangement / demerger /
    debenture-in-lieu-of-bonus events (deal-specific, no generic formula).
  - Every corporate action whose subject text contains a split/bonus keyword
    but that the regexes below fail to parse is written to
    `unparsed_actions.csv` instead of being silently dropped or
    mis-adjusted -- treat that file as a manual-review queue before trusting
    the adjusted series around those ISINs/dates.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Optional

import polars as pl
import requests

from src.data.nse_session import get_nse_session, nse_api_get

CORP_ACTIONS_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions"
    "?index=equities&from_date={frm}&to_date={to}"
)
CORP_ACTIONS_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-actions"

_SKIP_KEYWORDS = ("debenture", "demerger", "merger", "scheme of arrangement", "amalgamat")

_BONUS_RE = re.compile(r"bonus\D{0,15}?(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_SPLIT_FROM_RE = re.compile(r"from\s+rs\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_SPLIT_TO_RE = re.compile(r"to\s+r[se]\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_corporate_action_ratio(subject: str) -> Optional[float]:
    """
    Return the backward price-adjustment multiplier implied by `subject`,
    i.e. the factor to multiply all prices STRICTLY BEFORE the ex-date by,
    so they're on the same per-share basis as prices on/after the ex-date.
    Returns None if the subject isn't a mechanically-adjustable action, or
    mentions one but couldn't be parsed (caller should log these for review).
    """
    low = subject.lower()
    if any(kw in low for kw in _SKIP_KEYWORDS):
        return None

    multiplier = 1.0
    matched_anything = False

    bonus_m = _BONUS_RE.search(subject)
    if bonus_m:
        x, y = float(bonus_m.group(1)), float(bonus_m.group(2))
        if x > 0 and y > 0:
            multiplier *= y / (x + y)
            matched_anything = True

    from_m, to_m = _SPLIT_FROM_RE.search(subject), _SPLIT_TO_RE.search(subject)
    if from_m and to_m:
        old_face, new_face = float(from_m.group(1)), float(to_m.group(1))
        if old_face > 0 and new_face > 0:
            multiplier *= new_face / old_face
            matched_anything = True

    if not matched_anything:
        return None
    return multiplier


def fetch_corporate_actions(start_year: int, end_year: int, sleep_s: float = 0.5) -> pl.DataFrame:
    """Fetch raw corporate-action rows year by year (API is queried per calendar year to
    stay well under any implicit row cap on wide date ranges)."""
    import time

    session = get_nse_session()
    rows = []
    for year in range(start_year, end_year + 1):
        url = CORP_ACTIONS_URL.format(frm=f"01-01-{year}", to=f"31-12-{year}")
        try:
            data = nse_api_get(session, url, CORP_ACTIONS_REFERER)
        except RuntimeError as e:
            print(f"  [corporate_actions] {year}: {e}")
            continue
        for r in data:
            rows.append(r)
        time.sleep(sleep_s)

    if not rows:
        return pl.DataFrame(schema={
            "isin": pl.Utf8, "symbol": pl.Utf8, "series": pl.Utf8,
            "ex_date": pl.Date, "subject": pl.Utf8,
        })

    df = pl.DataFrame(rows)
    df = df.select(["isin", "symbol", "series", "exDate", "subject"]).rename({"exDate": "ex_date"})
    df = df.filter(pl.col("ex_date") != "-")
    df = df.with_columns(
        pl.col("ex_date").str.strip_chars().str.to_date("%d-%b-%Y", strict=False)
    ).drop_nulls("ex_date")
    return df


def build_adjustment_factors(corp_actions: pl.DataFrame, out_dir: Path | None = None) -> pl.DataFrame:
    """
    From raw corporate-action rows, compute a per-ISIN step function of
    cumulative backward adjustment factor: adj_factor(isin, date) applies to
    every price row with that isin strictly before the next action's ex_date.
    """
    actions = corp_actions.filter(pl.col("series") == "EQ").with_columns(
        pl.col("subject").map_elements(parse_corporate_action_ratio, return_dtype=pl.Float64).alias("multiplier")
    )

    unparsed = actions.filter(
        pl.col("multiplier").is_null()
        & pl.col("subject").str.to_lowercase().str.contains("bonus|split|sub-divi|sub divi|consolidat")
    )
    parsed = actions.filter(pl.col("multiplier").is_not_null())

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        unparsed.write_csv(out_dir / "unparsed_actions.csv")

    if parsed.height == 0:
        return pl.DataFrame(schema={"isin": pl.Utf8, "ex_date": pl.Date, "cum_factor": pl.Float64})

    # For each ISIN, sort actions by ex_date descending and take a reverse
    # cumulative product of multipliers -> cum_factor(ex_date) = product of
    # all multipliers for actions with ex_date' >= this ex_date.
    parsed = parsed.sort(["isin", "ex_date"], descending=[False, True])
    parsed = parsed.with_columns(
        pl.col("multiplier").cum_prod().over("isin").alias("cum_factor")
    )
    factors = parsed.select(["isin", "ex_date", "cum_factor"]).unique(subset=["isin", "ex_date"], keep="first")
    return factors.sort(["isin", "ex_date"])


def apply_adjustment(prices: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """
    Attach `close_adj`/`open_adj`/`high_adj`/`low_adj` columns to a prices
    frame (columns: isin, date, open, high, low, close, ...) using the
    step-function factors from build_adjustment_factors. A price row on
    `date` gets the smallest cum_factor among actions with ex_date > date
    (i.e. only actions strictly in its future affect it); rows on/after the
    last action need no adjustment (factor 1.0).
    """
    if factors.height == 0:
        return prices.with_columns([
            pl.col("open").alias("open_adj"),
            pl.col("high").alias("high_adj"),
            pl.col("low").alias("low_adj"),
            pl.col("close").alias("close_adj"),
            pl.lit(1.0).alias("adj_factor"),
        ])

    out_frames = []
    for isin, grp in prices.group_by("isin"):
        isin_val = isin[0] if isinstance(isin, tuple) else isin
        f = factors.filter(pl.col("isin") == isin_val).sort("ex_date")
        if f.height == 0:
            out_frames.append(grp.with_columns(pl.lit(1.0).alias("adj_factor")))
            continue
        ex_dates = f["ex_date"].to_list()
        cum_factors = f["cum_factor"].to_list()

        def factor_for(d):
            # smallest cum_factor among actions strictly after d;
            # equivalently: first ex_date > d walking from the earliest action.
            for ed, cf in zip(ex_dates, cum_factors):
                if ed > d:
                    return cf
            return 1.0

        grp = grp.with_columns(
            pl.col("date").map_elements(factor_for, return_dtype=pl.Float64).alias("adj_factor")
        )
        out_frames.append(grp)

    adjusted = pl.concat(out_frames)
    return adjusted.with_columns([
        (pl.col("open") * pl.col("adj_factor")).alias("open_adj"),
        (pl.col("high") * pl.col("adj_factor")).alias("high_adj"),
        (pl.col("low") * pl.col("adj_factor")).alias("low_adj"),
        (pl.col("close") * pl.col("adj_factor")).alias("close_adj"),
    ])
