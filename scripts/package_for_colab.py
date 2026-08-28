"""
Bundle this project for Colab. As of the GitHub remote
(https://github.com/yuvidewan/market_neural_net) existing, colab_train.ipynb
gets CODE via `git clone`/`git pull` -- the code bundle this script produces
is now a fallback (offline use, or testing local changes before pushing),
not the normal path. The DATA bundle is still needed every time: curated
data is intentionally gitignored (large, regenerable, not source), so it
still goes to Colab via a one-time Drive upload.

Produces:
  experiments/colab_code_bundle.zip  -- src/, scripts/, configs/, requirements.txt,
                                         pyproject.toml. Fallback only -- prefer
                                         pushing to GitHub and letting the
                                         notebook `git pull` instead.
  experiments/colab_data_bundle.zip  -- data/curated/ only (not raw/interim,
                                         those are much larger and not needed
                                         for training). ~300MB, upload once
                                         and only re-upload after a real
                                         re-ingest. THIS is the one you
                                         actually need regularly.

Usage:
    python -m scripts.package_for_colab --skip-data   # normal use: just refresh the code fallback, rare
    python -m scripts.package_for_colab                # first-time setup, or after a real data re-ingest
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_PATHS = ["src", "scripts", "configs", "requirements.txt", "pyproject.toml"]
DATA_PATHS = ["data/curated"]


def _zip_paths(paths: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in paths:
            src = ROOT / rel
            if src.is_file():
                zf.write(src, arcname=rel)
            elif src.is_dir():
                for f in src.rglob("*"):
                    if f.is_file() and "__pycache__" not in f.parts:
                        zf.write(f, arcname=str(f.relative_to(ROOT)))
            else:
                print(f"  [skip] {rel} not found")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path} ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments")
    ap.add_argument("--skip-data", action="store_true", help="only bundle code, not data/curated")
    args = ap.parse_args()

    print("Packaging code bundle ...")
    _zip_paths(CODE_PATHS, args.out_dir / "colab_code_bundle.zip")

    if not args.skip_data:
        print("Packaging data bundle (data/curated only) ...")
        _zip_paths(DATA_PATHS, args.out_dir / "colab_data_bundle.zip")
    else:
        print("Skipping data bundle (--skip-data)")

    print(f"\nUpload colab_data_bundle.zip to Google Drive at My Drive/market_neural_net/")
    print("(code now comes from GitHub via `git pull` inside the notebook -- the code zip is")
    print("just a fallback). Then open notebooks/colab_train.ipynb in Colab and run all cells.")


if __name__ == "__main__":
    main()
