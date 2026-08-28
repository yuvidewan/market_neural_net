"""
python -m scripts.build_universe
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.universe import build_universe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/curated/universe"))
    args = ap.parse_args()

    df = build_universe(args.out_dir)
    print(f"Fetched {df.height} (index, symbol) rows across indices: {sorted(df['index'].unique().to_list())}")


if __name__ == "__main__":
    main()
