"""
Bundle this project for Colab, per README §2.1: "Repo comes from git, not
from copy-paste" is the long-run goal once a GitHub remote exists, but there
isn't one yet -- so for now this makes two zip files to upload to Google
Drive instead. Once a remote exists, notebooks/colab_train.ipynb's git-clone
cell can be swapped in and this script becomes unnecessary.

Produces:
  experiments/colab_code_bundle.zip  -- src/, scripts/, configs/, requirements.txt,
                                         pyproject.toml. Small, re-run this and
                                         re-upload whenever the code changes.
  experiments/colab_data_bundle.zip  -- data/curated/ only (not raw/interim,
                                         those are much larger and not needed
                                         for training). ~300MB, upload once
                                         and only re-upload after a real
                                         re-ingest.

Usage:
    python -m scripts.package_for_colab
    python -m scripts.package_for_colab --skip-data   # code bundle only
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

    print(f"\nUpload both zips to Google Drive, e.g. My Drive/market_neural_net/")
    print("then open notebooks/colab_train.ipynb in Colab and run all cells.")


if __name__ == "__main__":
    main()
