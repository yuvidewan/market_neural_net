"""
python -m scripts.build_curated
python -m scripts.build_curated --no-fetch-actions   # skip the live NSE corp-actions call (offline/dev)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_curated import build_curated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interim-dir", type=Path, default=Path("data/interim/bhavcopy"))
    ap.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    ap.add_argument("--no-fetch-actions", action="store_true")
    ap.add_argument("--corp-actions-start-year", type=int, default=1996)
    args = ap.parse_args()

    stats = build_curated(
        args.interim_dir, args.curated_dir,
        fetch_actions=not args.no_fetch_actions,
        corp_actions_start_year=args.corp_actions_start_year,
    )
    print("Done:", stats)


if __name__ == "__main__":
    main()
