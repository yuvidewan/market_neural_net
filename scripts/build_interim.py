"""
python -m scripts.build_interim
python -m scripts.build_interim --raw-dir data/raw/bhavcopy --interim-dir data/interim/bhavcopy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest.build_interim import build_interim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/bhavcopy"))
    ap.add_argument("--interim-dir", type=Path, default=Path("data/interim/bhavcopy"))
    args = ap.parse_args()

    stats = build_interim(args.raw_dir, args.interim_dir)
    print("Done:", {k: v for k, v in stats.items() if k != "parse_errors"})
    if stats["parse_errors"]:
        print(f"{len(stats['parse_errors'])} parse errors, first 10:")
        for name, err in stats["parse_errors"][:10]:
            print(" ", name, err)


if __name__ == "__main__":
    main()
