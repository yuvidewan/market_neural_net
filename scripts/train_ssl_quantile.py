"""
M3: pretrain a causal TCN via the self-supervised quantile-regression
objective (README Phase 3, objective #2), then evaluate it out-of-sample
with the cross-sectional rank IC metric -- the actual M3 gate ("OOS rank IC
> 0.02, stable sign across folds").

Scope decision, learned from M2's 6.5-hour CPU surprise: TCN convolutions
are parallel across time (no LSTM-style sequential bottleneck), so this
should be genuinely CPU-tractable at meaningfully larger scale than the
LSTM baseline was. Still starts conservative -- default universe size
matches M2's for a clean before/after comparison -- with `--n-symbols`
there to scale up once the runtime is confirmed reasonable.

Median quantile (tau=0.5) is used wherever a single point prediction is
needed (rank IC, the M2-comparable sign-backtest) -- see quantile.py's
documented non-crossing-quantiles caveat for why the full distribution
isn't trusted as a calibrated confidence signal yet.

Checkpoint/resume (per-epoch, per-fold -- see src/train/checkpointing.py):
safe to Ctrl-C or lose a Colab session mid-run and just re-run the exact same
command with the same --out-dir; already-completed folds are skipped, and an
interrupted fold resumes from its last saved epoch. Pass --fresh to ignore
any existing checkpoint and start over.

IMPORTANT for Colab: keep --out-dir on LOCAL disk and pass --mirror-dir
pointing at your mounted Drive instead of putting --out-dir on Drive
directly -- Drive's FUSE mount doesn't handle interrupted writes reliably
enough to be the primary read/write target; local-first + best-effort
mirror survives a full VM recycle without that risk (see checkpointing.py).

Usage:
    python -u -m scripts.train_ssl_quantile
    python -u -m scripts.train_ssl_quantile --n-symbols 40 --epochs 6
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
from src.eval.metrics import rank_ic_summary, sign_backtest
from src.eval.walkforward import assert_no_leakage, generate_yearly_folds, split_masks
from src.models.encoders.tcn import TCNEncoder
from src.models.ssl.quantile import EncoderQuantileWrapper, QUANTILES, pinball_loss
from src.train.checkpointing import FoldCheckpointer
from scripts.train_lstm_baseline import scan_curated_prices, select_liquid_universe


def train_one_fold(train_ds, model, epochs, batch_size, lr, device, checkpointer, fold_label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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
            loss = pinball_loss(pred, yb, model.head.quantiles)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"    epoch {epoch + 1}/{epochs}  train_pinball={total_loss / max(n_batches, 1):.6f}", flush=True)
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
    preds = np.concatenate(preds)          # [N, n_quantiles]
    actuals = np.concatenate(actuals)      # [N]
    dates = np.array([s.label_date for s in test_ds.samples])
    return preds, actuals, dates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=120)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embargo-days", type=int, default=10)
    ap.add_argument("--test-years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/ssl_quantile_tcn"))
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
    print(f"  {subset.height} rows, date range {subset['date'].min()} -> {subset['date'].max()}", flush=True)

    print("Computing causal features ...", flush=True)
    features = compute_features(subset)

    print(f"Building sequence dataset (seq_len={args.seq_len}) ...", flush=True)
    ds = SequenceDataset(features, seq_len=args.seq_len)
    print(f"  {len(ds)} valid windows", flush=True)

    folds = generate_yearly_folds(args.test_years)
    completed_by_label = {r["fold"]: r for r in checkpointer.completed_results()}
    all_results = []
    median_idx = QUANTILES.index(0.5)

    for fold in folds:
        if fold.label in completed_by_label:
            print(f"\n=== Fold {fold.label}: already completed (from checkpoint), skipping ===", flush=True)
            all_results.append(completed_by_label[fold.label])
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
        assert_no_leakage(train_ds, fold, embargo_days=args.embargo_days)

        encoder = TCNEncoder(n_features=len(FEATURE_COLUMNS), channels=args.channels)
        model = EncoderQuantileWrapper(encoder, hidden_size=args.channels).to(device)

        train_one_fold(train_ds, model, args.epochs, args.batch_size, args.lr, device, checkpointer, fold.label)
        preds, actuals, dates = evaluate_fold(test_ds, model, args.batch_size, device)
        median_pred = preds[:, median_idx]

        ic = rank_ic_summary(dates, median_pred, actuals)
        bt = sign_backtest(median_pred, actuals, cost_bps=10.0)
        bt.pop("net_pnl")

        print(f"  mean_ic={ic['mean_ic']:.4f}  ic_ir={ic['ic_ir']:.3f}  "
              f"pct_positive_days={ic['pct_positive_days']:.2f}  n_days={ic['n_days']}", flush=True)
        print(f"  [sign-backtest, comparable to M2] gross_sharpe={bt['gross_sharpe']:.3f}  "
              f"net_sharpe={bt['net_sharpe']:.3f}  hit_rate={bt['hit_rate']:.3f}", flush=True)

        fold_result = {"fold": fold.label, "rank_ic": ic, "sign_backtest": bt}
        checkpointer.mark_fold_complete(fold.label, fold_result)
        all_results.append(fold_result)

    if all_results:
        mean_ics = [r["rank_ic"]["mean_ic"] for r in all_results if not np.isnan(r["rank_ic"]["mean_ic"])]
        overall_mean_ic = float(np.mean(mean_ics)) if mean_ics else float("nan")
        all_positive = all(m > 0 for m in mean_ics) if mean_ics else False
        all_negative = all(m < 0 for m in mean_ics) if mean_ics else False
        stable_sign = all_positive or all_negative

        print(f"\n=== Overall ===", flush=True)
        print(f"  per-fold mean IC: {[round(m, 4) for m in mean_ics]}", flush=True)
        print(f"  overall mean IC: {overall_mean_ic:.4f}   stable sign across folds: {stable_sign}", flush=True)
        gate = overall_mean_ic > 0.02 and stable_sign
        print(f"  M3 GATE (mean IC > 0.02 AND stable sign): {'PASS' if gate else 'FAIL'}", flush=True)

        report = {
            "universe_size": len(universe), "n_samples_total": len(ds),
            "seq_len": args.seq_len, "channels": args.channels, "epochs": args.epochs,
            "folds": all_results, "overall_mean_ic": overall_mean_ic,
            "stable_sign": stable_sign, "m3_gate_pass": gate,
            "generated_at": dt.datetime.now().isoformat(),
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport written to {args.out_dir / 'report.json'}", flush=True)
    else:
        print("\nNo folds produced results.", flush=True)


if __name__ == "__main__":
    main()
