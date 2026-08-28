"""
NSE EOD bhavcopy ingestion.

Pulls the exchange's own end-of-day cash-market bhavcopy files directly from
NSE's public archive (nsearchives.nseindia.com). No login, no scraping tricks
beyond a browser User-Agent header — this is the same file NSE publishes for
manual download, just automated and made resumable.

Two file formats exist depending on date, because NSE changed formats in
2024:

  - "old" format  (roughly 1998 -> early Jul 2024):
      https://nsearchives.nseindia.com/content/historical/EQUITIES/
        {YYYY}/{MON}/cm{DD}{MON}{YYYY}bhav.csv.zip
  - "new" UDIFF format (roughly Jul 2024 -> present):
      https://nsearchives.nseindia.com/content/cm/
        BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip

Rather than hardcode the exact cutover date (NSE has moved it before and
could again), both candidate URLs are tried for every date, ordered by which
one is expected to work first. A 404 on every candidate means "not a trading
day" (weekend/holiday) and is cached so we never re-request it.
"""

from __future__ import annotations

import datetime as dt
import io
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Date on/after which the new UDIFF format is tried first. This is just an
# ordering hint for efficiency (saves one wasted request per day) -- both
# formats are always attempted as fallback, so an inexact boundary is safe.
FORMAT_CUTOVER = dt.date(2024, 7, 8)

MONTH_ABBR = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def old_format_url(date: dt.date) -> str:
    mon = MONTH_ABBR[date.month - 1]
    return (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
        f"{date.year}/{mon}/cm{date.day:02d}{mon}{date.year}bhav.csv.zip"
    )


def new_format_url(date: dt.date) -> str:
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{date.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def candidate_urls(date: dt.date) -> list[str]:
    old, new = old_format_url(date), new_format_url(date)
    return [new, old] if date >= FORMAT_CUTOVER else [old, new]


@dataclass
class FetchResult:
    date: dt.date
    status: str  # "ok" | "holiday" | "error"
    content: bytes | None = None
    url: str | None = None
    error: str | None = None


def fetch_bhavcopy(
    date: dt.date,
    session: requests.Session,
    timeout: float = 15.0,
    max_retries: int = 3,
) -> FetchResult:
    """Fetch one day's raw bhavcopy zip bytes. Tries both URL formats."""
    last_error = None
    for url in candidate_urls(date):
        for attempt in range(max_retries):
            try:
                r = session.get(url, timeout=timeout)
            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 200 and r.content:
                return FetchResult(date=date, status="ok", content=r.content, url=url)
            if r.status_code == 404:
                last_error = "404"
                break  # try the other URL format, no point retrying a 404
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}"
                time.sleep(1.5 * (attempt + 1))
                continue
            last_error = f"HTTP {r.status_code}"
            break
    if last_error == "404":
        return FetchResult(date=date, status="holiday", error="404 on all formats")
    return FetchResult(date=date, status="error", error=last_error)


# --------------------------------------------------------------------------
# Parsing: raw zip bytes -> normalized polars DataFrame
# --------------------------------------------------------------------------

_OLD_COLS = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LAST": "last",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TOTALTRADES": "n_trades",
    "ISIN": "isin",
}

_NEW_COLS = {
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "n_trades",
    "ISIN": "isin",
    "FinInstrmTp": "instrument_type",
}

_OUT_SCHEMA_ORDER = [
    "date", "isin", "symbol", "series",
    "open", "high", "low", "close", "prev_close", "last",
    "volume", "turnover", "n_trades",
]


def parse_bhavcopy_zip(content: bytes, date: dt.date) -> pl.DataFrame:
    """Parse one day's raw zip bytes (either format) into a normalized frame."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        inner_name = z.namelist()[0]
        raw = z.read(inner_name)

    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)  # everything as str first
    df.columns = [c.strip() for c in df.columns]

    is_new_format = "TckrSymb" in df.columns
    colmap = _NEW_COLS if is_new_format else _OLD_COLS

    present = {k: v for k, v in colmap.items() if k in df.columns}
    df = df.select(list(present.keys())).rename(present)

    if "instrument_type" in df.columns:
        # New UDIFF CM-segment file includes non-equity instrument types
        # (bonds/T-bills/SGBs are FinInstrmTp != "STK"). Keep only equities
        # here; other series filtering (EQ vs BE/BZ/...) happens downstream.
        df = df.filter(pl.col("instrument_type") == "STK").drop("instrument_type")

    numeric_cols = ["open", "high", "low", "close", "prev_close", "last",
                     "volume", "turnover", "n_trades"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.with_columns(
                pl.col(c).str.strip_chars().replace("", None).cast(pl.Float64, strict=False)
            )

    df = df.with_columns(pl.lit(date).alias("date"))
    # Fields absent from a given format (e.g. no ISIN/TOTALTRADES in the
    # pre-2000s files) are filled as typed nulls -- untyped `lit(None)`
    # produces a Null-dtype column that later string/int ops reject.
    string_cols = {"isin", "symbol", "series"}
    float_cols = {"open", "high", "low", "close", "prev_close", "last", "volume", "turnover"}
    for c in _OUT_SCHEMA_ORDER:
        if c in df.columns:
            continue
        if c in string_cols:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
        elif c in float_cols:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        elif c == "n_trades":
            df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias(c))
        else:
            df = df.with_columns(pl.lit(None).alias(c))

    df = df.select(_OUT_SCHEMA_ORDER)
    df = df.with_columns([
        pl.col("symbol").str.strip_chars(),
        pl.col("series").str.strip_chars(),
        pl.col("isin").str.strip_chars(),
        pl.col("n_trades").cast(pl.Int64, strict=False),
    ])
    return df


# --------------------------------------------------------------------------
# Resumable range downloader
# --------------------------------------------------------------------------

def _year_dir(raw_dir: Path, date: dt.date) -> Path:
    d = raw_dir / str(date.year)
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_range(
    start: dt.date,
    end: dt.date,
    raw_dir: Path,
    sleep_range: tuple[float, float] = (0.25, 0.6),
    progress: bool = True,
) -> dict:
    """
    Download every trading day's raw bhavcopy zip in [start, end] into
    raw_dir/{year}/{date}.zip. Idempotent and resumable:
      - a zip already on disk is skipped (no request made)
      - a date already recorded as a holiday is skipped (no request made)
      - transient errors are retried within fetch_bhavcopy, and if still
        failing are logged and simply retried again the next run
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    holidays_path = raw_dir / "_holidays.txt"
    errors_path = raw_dir / "_errors.txt"

    known_holidays = set()
    if holidays_path.exists():
        known_holidays = {
            dt.date.fromisoformat(l.strip())
            for l in holidays_path.read_text().splitlines() if l.strip()
        }

    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri only; NSE never trades weekends
            dates.append(d)
        d += dt.timedelta(days=1)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    stats = {"ok": 0, "skipped_existing": 0, "skipped_holiday": 0, "holiday": 0, "error": 0}

    iterator = dates
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(dates, desc="bhavcopy")
        except ImportError:
            pass

    for date in iterator:
        target = _year_dir(raw_dir, date) / f"{date.isoformat()}.zip"
        if target.exists():
            stats["skipped_existing"] += 1
            continue
        if date in known_holidays:
            stats["skipped_holiday"] += 1
            continue

        result = fetch_bhavcopy(date, session)
        if result.status == "ok":
            target.write_bytes(result.content)
            stats["ok"] += 1
        elif result.status == "holiday":
            with holidays_path.open("a") as f:
                f.write(date.isoformat() + "\n")
            stats["holiday"] += 1
        else:
            with errors_path.open("a") as f:
                f.write(f"{date.isoformat()}\t{result.error}\n")
            stats["error"] += 1

        time.sleep(random.uniform(*sleep_range))

    return stats
