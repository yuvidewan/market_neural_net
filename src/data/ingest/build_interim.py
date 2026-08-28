"""
Parse every raw bhavcopy zip into normalized, year-partitioned parquet under
data/interim/bhavcopy/. This is a pure re-run-able transform of data/raw/ --
never edits raw files, safe to delete and rebuild at any time.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from src.data.ingest.bhavcopy import parse_bhavcopy_zip


def build_interim(raw_dir: Path, interim_dir: Path, progress: bool = True) -> dict:
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.isdigit())
    stats = {"years": 0, "days": 0, "rows": 0, "parse_errors": []}

    for year_dir in year_dirs:
        zips = sorted(year_dir.glob("*.zip"))
        if not zips:
            continue
        frames = []
        iterator = zips
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(zips, desc=f"parse {year_dir.name}")
            except ImportError:
                pass
        for zpath in iterator:
            date = dt.date.fromisoformat(zpath.stem)
            try:
                df = parse_bhavcopy_zip(zpath.read_bytes(), date)
            except Exception as e:  # noqa: BLE001 - log and continue, one bad day shouldn't kill the run
                stats["parse_errors"].append((zpath.name, str(e)))
                continue
            frames.append(df)
            stats["days"] += 1

        if not frames:
            continue
        year_df = pl.concat(frames)
        out_path = interim_dir / f"{year_dir.name}.parquet"
        year_df.write_parquet(out_path)
        stats["years"] += 1
        stats["rows"] += year_df.height

    return stats
