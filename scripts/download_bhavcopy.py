"""
Resumable downloader for NSE daily bhavcopy archives.

Usage:
    python -m scripts.download_bhavcopy --start 2000-01-01 --end 2026-08-28
    python -m scripts.download_bhavcopy --start 2015-01-01 --end 2026-08-28 --raw-dir data/raw/bhavcopy

Safe to Ctrl-C and re-run: already-downloaded days and confirmed holidays are
skipped without a network request.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest.bhavcopy import download_range


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=parse_date, required=True)
    ap.add_argument("--end", type=parse_date, required=True)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/bhavcopy"))
    ap.add_argument("--min-sleep", type=float, default=0.25)
    ap.add_argument("--max-sleep", type=float, default=0.6)
    args = ap.parse_args()

    print(f"Downloading bhavcopy {args.start} -> {args.end} into {args.raw_dir}")
    stats = download_range(
        args.start, args.end, args.raw_dir,
        sleep_range=(args.min_sleep, args.max_sleep),
    )
    print("Done:", stats)


if __name__ == "__main__":
    main()
