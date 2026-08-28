"""
python -m scripts.run_quality_report
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.quality_report import generate_quality_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    args = ap.parse_args()

    summary = generate_quality_report(args.curated_dir)
    print("=== Data Quality Report ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
