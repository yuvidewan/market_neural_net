"""
v2 (README Phase 3): pretrain the two-axis causal Transformer via the same
self-supervised quantile-regression objective as M3's TCN run, then evaluate
with the same cross-sectional rank IC gate -- this is meant as a direct,
apples-to-apples comparison against v1 (train_ssl_quantile.py), same
universe-selection logic, same folds, same metrics, same gate. The only
things that differ are the encoder (two-axis Transformer vs. TCN) and the
dataset (PanelSequenceDataset, one cross-sectional panel per date, vs.
SequenceDataset's independent per-symbol windows) -- because the
transformer's cross-sectional attention needs a whole panel per forward pass
(see src/models/encoders/transformer.py and src/data/panel_dataset.py).

One consequence of panels varying in size (not every symbol has valid data
every date): there's no fixed-shape mini-batch to hand a DataLoader's default
collate_fn, so this script iterates panels directly, one per optimizer step,
shuffled each epoch. Each panel already contains up to n_symbols rows to
average the loss over (often 100-200), a comparable magnitude to the
batch_size=128 used elsewhere in this repo, so this isn't a noisier gradient
signal than the LSTM/TCN scripts get -- just organized around dates instead
of independent samples.

Checkpoint/resume (per-epoch, per-fold -- see src/train/checkpointing.py):
this script does far more optimizer steps per epoch than the batch-based
scripts (one per calendar date rather than one per fixed-size batch), so
it's the one most likely to actually need this. Safe to Ctrl-C or lose a
Colab session mid-run and re-run the exact same command with the same
--out-dir; already-completed folds are skipped, interrupted folds resume
from their last saved epoch. Pass --fresh to start over.

IMPORTANT for Colab: keep --out-dir on LOCAL disk and pass --mirror-dir
pointing at your mounted Drive instead of putting --out-dir on Drive
directly -- Drive's FUSE mount doesn't handle interrupted writes reliably
enough to be the primary read/write target; local-first + best-effort
mirror survives a full VM recycle without that risk (see checkpointing.py).

Usage:
    python -u -m scripts.train_transformer_ssl
    python -u -m scripts.train_transformer_ssl --n-symbols 200 --epochs 8
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

from src.data.features import FEATURE_COLUMNS, compute_features
from src.data.panel_dataset import PanelSequenceDataset
from src.eval.metrics import rank_ic_summary, sign_backtest
from src.eval.walkforward import assert_no_leakage, generate_yearly_folds, split_masks
from src.models.encoders.transformer import TwoAxisTransformerEncoder
from src.models.ssl.quantile import EncoderQuantileWrapper, QUANTILES, pinball_loss
from src.train.checkpointing import FoldCheckpointer
from scripts.train_lstm_baseline import scan_curated_prices, select_liquid_universe


def train_one_fold(train_ds, model, epochs, lr, device, seed, checkpointer, fold_label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    start_epoch = checkpointer.resume_epoch_for(fold_label)
    if start_epoch > 0:
        if checkpointer.load_model_state(fold_label, model, opt, device):
            print(f"    resumed from checkpoint at epoch {start_epoch}", flush=True)
        else:
            start_epoch = 0

    model.train()
    n = len(train_ds)
    for epoch in range(start_epoch, epochs):
        order = rng.permutation(n)
        # LR warmup over the first 10% of steps of epoch 0 ONLY (0.05x -> 1.0x lr):
        # each step here is one full cross-sectional panel, not a small mini-batch,
        # so gradient variance step-to-step is high, and this is a fairly deep
        # (8-block, 8-head) attention stack with randomly-initialized attention
        # weights -- exactly the combination that collapses to a constant
        # prediction without warmup (confirmed: the first real run did this,
        # every fold's rank IC came back NaN/0-days, sign_backtest turnover
        # ~1000x lower than the TCN's, hit rate ~50% -- the signature of a model
        # that gave up and started predicting the same value regardless of input).
        # A resumed run skips epoch 0's warmup as already-done, which is correct.
        warmup_steps = max(1, n // 10) if epoch == 0 else 0
        total_loss, n_steps = 0.0, 0
        for step_in_epoch, i in enumerate(order):
            if step_in_epoch < warmup_steps:
                warmup_lr = lr * (0.05 + 0.95 * (step_in_epoch + 1) / warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = warmup_lr
            elif epoch == 0 and step_in_epoch == warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = lr  # warmup done, lock in the target LR for the rest of training

            xb, yb = train_ds[i]
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = pinball_loss(pred, yb, model.head.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            n_steps += 1
        print(f"    epoch {epoch + 1}/{epochs}  train_pinball={total_loss / max(n_steps, 1):.6f}", flush=True)
        checkpointer.save_epoch(fold_label, epoch + 1, model, opt)


def evaluate_fold(test_ds, model, device):
    model.eval()
    all_dates, all_preds, all_actuals = [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            xb, yb = test_ds[i]
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()          # [n_symbols_that_date, n_quantiles]
            label_date = test_ds.samples[i].label_date
            all_dates.extend([label_date] * len(yb))
            all_preds.append(pred)
            all_actuals.append(yb.numpy())
    preds = np.concatenate(all_preds) if all_preds else np.empty((0, len(QUANTILES)))
    actuals = np.concatenate(all_actuals) if all_actuals else np.empty((0,))
    return preds, actuals, np.array(all_dates)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    ap.add_argument("--n-symbols", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=120)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)  # lower than M2/M3's 1e-3 -- see train_one_fold's warmup comment
    ap.add_argument("--min-symbols-per-date", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=10)
    ap.add_argument("--test-years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/ssl_quantile_transformer"))
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

    print(f"Building panel dataset (seq_len={args.seq_len}) ...", flush=True)
    ds = PanelSequenceDataset(
        features, universe, seq_len=args.seq_len, min_symbols_per_date=args.min_symbols_per_date,
    )
    print(f"  {len(ds)} valid panel-dates", flush=True)

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
        print(f"  train panel-dates: {n_train}   test panel-dates: {n_test}", flush=True)
        if n_train < 50 or n_test < 5:
            print("  skipping fold (insufficient panel-dates)", flush=True)
            continue

        train_ds = ds.subset_by_mask(train_mask)
        test_ds = ds.subset_by_mask(test_mask)
        assert_no_leakage(train_ds, fold, embargo_days=args.embargo_days)

        encoder = TwoAxisTransformerEncoder(
            n_features=len(FEATURE_COLUMNS), d_model=args.d_model, n_heads=args.n_heads,
            n_blocks=args.n_blocks, patch_size=args.patch_size,
        )
        model = EncoderQuantileWrapper(encoder, hidden_size=args.d_model).to(device)

        train_one_fold(train_ds, model, args.epochs, args.lr, device, args.seed, checkpointer, fold.label)
        preds, actuals, dates = evaluate_fold(test_ds, model, device)
        if len(actuals) == 0:
            print("  no test predictions produced, skipping fold metrics", flush=True)
            continue
        median_pred = preds[:, median_idx]

        ic = rank_ic_summary(dates, median_pred, actuals)
        bt = sign_backtest(median_pred, actuals, cost_bps=10.0)
        bt.pop("net_pnl")

        if ic["n_days"] == 0:
            # rank_ic_by_date skips a date if predictions have zero variance
            # across symbols that day. Zero valid days across an entire fold
            # means EVERY date's predictions were constant -- almost always a
            # collapsed model (predicting the same value regardless of input),
            # not a data problem. Printed live so this is caught during the
            # run, not discovered later by downloading and reading report.json.
            pred_std = float(np.std(median_pred))
            print(f"  WARNING: 0 valid rank-IC days -- predictions likely collapsed to "
                  f"near-constant (overall pred std={pred_std:.6g}, "
                  f"lr/warmup/grad-clip may need adjusting)", flush=True)
        print(f"  mean_ic={ic['mean_ic']:.4f}  ic_ir={ic['ic_ir']:.3f}  "
              f"pct_positive_days={ic['pct_positive_days']:.2f}  n_days={ic['n_days']}", flush=True)
        print(f"  [sign-backtest, comparable to M2/M3] gross_sharpe={bt['gross_sharpe']:.3f}  "
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
        print(f"  v2 GATE (mean IC > 0.02 AND stable sign, same bar as M3): {'PASS' if gate else 'FAIL'}", flush=True)

        report = {
            "universe_size": len(universe), "n_panel_dates_total": len(ds),
            "seq_len": args.seq_len, "d_model": args.d_model, "n_heads": args.n_heads,
            "n_blocks": args.n_blocks, "patch_size": args.patch_size, "epochs": args.epochs,
            "folds": all_results, "overall_mean_ic": overall_mean_ic,
            "stable_sign": stable_sign, "gate_pass": gate,
            "generated_at": dt.datetime.now().isoformat(),
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport written to {args.out_dir / 'report.json'}", flush=True)
    else:
        print("\nNo folds produced results.", flush=True)


if __name__ == "__main__":
    main()
