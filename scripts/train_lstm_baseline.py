"""
M2 deliverable: train the v0 LSTM baseline on real curated data and run it
through the purged/embargoed walk-forward harness end to end.

Universe simplification, stated explicitly: this picks the top-N most liquid
ISINs by trailing average daily turnover as of the most recent date in the
dataset, then uses each one's full available history. That is NOT a
point-in-time universe (a name liquid today might not have been liquid in
2015) -- it's a deliberate simplification to get a first real baseline
running quickly. Phase 5's real backtest must reselect the universe
point-in-time per fold; tracked as follow-up, not silently assumed fixed.

Checkpoint/resume (per-epoch, per-fold -- see src/train/checkpointing.py):
safe to Ctrl-C or lose a Colab session mid-run and just re-run the exact same
command with the same --out-dir; already-completed folds are skipped, and an
interrupted fold resumes from its last saved epoch, not from scratch. Pass
--fresh to ignore any existing checkpoint and start over (e.g. after a real
architecture change makes the old one incompatible).

IMPORTANT for Colab: keep --out-dir on LOCAL disk (the default, e.g.
experiments/...) and pass --mirror-dir pointing at your mounted Drive
instead of putting --out-dir on Drive directly. An earlier version of this
had --out-dir write straight to Drive, which turned out to cause the exact
data loss it was meant to prevent -- Drive's FUSE mount doesn't handle
interrupted writes reliably, and a checkpoint write cut off by a disconnect
could corrupt the resume state itself. Local-first + best-effort mirror
avoids that while still surviving a full VM recycle (mirror gets hydrated
back to a fresh local out-dir automatically).

Usage:
    python -m scripts.train_lstm_baseline
    python -m scripts.train_lstm_baseline --n-symbols 40 --epochs 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SequenceDataset
from src.data.features import FEATURE_COLUMNS, compute_features
from src.data.universe import liquid_universe_from_prices
from src.eval.metrics import sign_backtest
from src.eval.walkforward import Fold, assert_no_leakage, generate_yearly_folds, split_masks
from src.models.encoders.lstm import LSTMBaseline
from src.train.checkpointing import FoldCheckpointer


def scan_curated_prices(curated_dir: Path) -> pl.LazyFrame:
    """Lazy scan, not an eager read -- selecting a 40-symbol subset out of
    5,287 shouldn't require materializing all 9.2M rows first. Filters get
    pushed down before `.collect()` is ever called."""
    pattern = str(Path(curated_dir) / "prices" / "year=*" / "part.parquet")
    return pl.scan_parquet(pattern)


def select_liquid_universe(lazy_prices: pl.LazyFrame, n_symbols: int, lookback_days: int = 365) -> tuple[list[str], object]:
    """Determines liquidity from only a recent window (collected small), not
    the full history -- ADV over the last ~1.5x lookback_days is enough."""
    as_of = lazy_prices.select(pl.col("date").max()).collect().item()
    window_start = as_of - dt.timedelta(days=int(lookback_days * 1.6))
    recent = (
        lazy_prices
        .filter((pl.col("date") <= as_of) & (pl.col("date") > window_start) & (pl.col("series") == "EQ"))
        .select(["isin", "symbol", "series", "date", "turnover"])
        .collect()
    )
    liquid = liquid_universe_from_prices(recent, as_of_date=as_of, lookback_days=lookback_days, min_adv_inr=1e7)
    top = liquid.sort("adv_inr", descending=True).head(n_symbols)
    return top["isin"].to_list(), as_of


def train_one_fold(train_ds, model, epochs, batch_size, lr, device, checkpointer, fold_label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    start_epoch = checkpointer.resume_epoch_for(fold_label)
    if start_epoch > 0:
        if checkpointer.load_model_state(fold_label, model, opt, device):
            print(f"    resumed from checkpoint at epoch {start_epoch}", flush=True)
        else:
            start_epoch = 0

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"    epoch {epoch + 1}/{epochs}  train_mse={total_loss / max(n_batches, 1):.6f}", flush=True)
        checkpointer.save_epoch(fold_label, epoch + 1, model, opt)


def evaluate_fold(test_ds, model, batch_size, device):
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(actuals)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=120)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embargo-days", type=int, default=10)
    ap.add_argument("--test-years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/lstm_baseline"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint in --out-dir and start over")
    ap.add_argument("--mirror-dir", type=Path, default=None,
                     help="best-effort backup copy of checkpoints (e.g. a Drive-mounted path in Colab) -- "
                          "--out-dir itself should stay LOCAL disk for speed/reliability; a Drive hiccup "
                          "here only skips that mirror write, it never crashes training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = FoldCheckpointer(args.out_dir, run_args=vars(args), fresh=args.fresh, mirror_dir=args.mirror_dir)

    print("Scanning curated prices (lazy) ...", flush=True)
    lazy_prices = scan_curated_prices(args.curated_dir)

    universe, as_of = select_liquid_universe(lazy_prices, n_symbols=args.n_symbols)
    print(f"Selected {len(universe)} liquid ISINs (top by trailing ADV as of {as_of})", flush=True)

    subset = lazy_prices.filter(pl.col("isin").is_in(universe)).collect()
    print(f"  {subset.height} rows in liquid subset, "
          f"date range {subset['date'].min()} -> {subset['date'].max()}", flush=True)

    print("Computing causal features ...", flush=True)
    features = compute_features(subset)

    print(f"Building sequence dataset (seq_len={args.seq_len}) ...", flush=True)
    ds = SequenceDataset(features, seq_len=args.seq_len)
    print(f"  {len(ds)} valid windows", flush=True)

    folds = generate_yearly_folds(args.test_years)
    completed_by_label = {r["fold"]: r for r in checkpointer.completed_results()}
    all_results = []
    all_net_pnl = []

    for fold in folds:
        if fold.label in completed_by_label:
            saved = completed_by_label[fold.label]
            print(f"\n=== Fold {fold.label}: already completed (from checkpoint), skipping ===", flush=True)
            all_net_pnl.append(np.array(saved["net_pnl"]))
            all_results.append({k: v for k, v in saved.items() if k != "net_pnl"})
            continue

        print(f"\n=== Fold {fold.label}: test [{fold.test_start} .. {fold.test_end}] "
              f"(embargo {args.embargo_days}d) ===", flush=True)
        train_mask, test_mask = split_masks(ds, fold, embargo_days=args.embargo_days)
        n_train, n_test = train_mask.sum(), test_mask.sum()
        print(f"  train samples: {n_train}   test samples: {n_test}", flush=True)
        if n_train < 200 or n_test < 20:
            print("  skipping fold (insufficient samples)", flush=True)
            continue

        train_ds = ds.subset_by_mask(train_mask)
        test_ds = ds.subset_by_mask(test_mask)
        assert_no_leakage(train_ds, fold, embargo_days=args.embargo_days)  # hard runtime guard

        model = LSTMBaseline(
            n_features=len(FEATURE_COLUMNS),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ).to(device)

        train_one_fold(train_ds, model, args.epochs, args.batch_size, args.lr, device, checkpointer, fold.label)
        preds, actuals = evaluate_fold(test_ds, model, args.batch_size, device)
        result = sign_backtest(preds, actuals, cost_bps=10.0)

        print(f"  gross_sharpe={result['gross_sharpe']:.3f}  net_sharpe={result['net_sharpe']:.3f}  "
              f"hit_rate={result['hit_rate']:.3f}  avg_turnover/day={result['avg_turnover_per_day']:.3f}", flush=True)

        full_result = {"fold": fold.label, **result}
        full_result["net_pnl"] = result["net_pnl"].tolist()  # kept for resume, stripped from the final report below
        checkpointer.mark_fold_complete(fold.label, full_result)

        all_net_pnl.append(result.pop("net_pnl"))
        all_results.append({"fold": fold.label, **result})

    if all_net_pnl:
        combined = np.concatenate(all_net_pnl)
        overall_sharpe = float(np.sqrt(252) * combined.mean() / (combined.std(ddof=1) + 1e-12))
        print(f"\n=== Overall (all folds concatenated) ===", flush=True)
        print(f"  n_days={len(combined)}  net_sharpe={overall_sharpe:.3f}  "
              f"mean_daily_ret={combined.mean():.6f}", flush=True)

        report = {
            "universe_size": len(universe),
            "n_samples_total": len(ds),
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "folds": all_results,
            "overall_net_sharpe": overall_sharpe,
            "overall_n_days": len(combined),
            "generated_at": dt.datetime.now().isoformat(),
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.out_dir / 'report.json'}", flush=True)
    else:
        print("\nNo folds produced results.", flush=True)


if __name__ == "__main__":
    main()
