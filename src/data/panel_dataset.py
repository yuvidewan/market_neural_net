"""
Cross-sectional panel dataset for the v2 transformer (README Phase 3).

SequenceDataset (dataset.py) gives independent (isin, end_date) samples --
fine for the LSTM/TCN, which process one symbol at a time and only get
cross-sectional information as a post-hoc evaluation grouping (rank_ic
groups predictions by date AFTER they're made). The two-axis transformer's
cross-sectional attention needs the opposite: every symbol as of the SAME
date handed to the model in ONE forward pass, so the model itself can mix
information across names during training and inference, not just at
evaluation time.

A "sample" here is therefore one calendar date: the trailing seq_len window
for every symbol in the universe that has valid data on that date, stacked
into one panel. Symbols without enough history yet (recent IPO), or missing
that exact date, or with a NaN feature/target, are simply excluded from that
date's panel -- panels vary in size (n_symbols_that_date), which is why
training iterates one panel per step rather than through a shape-stacking
DataLoader collate (see scripts/train_transformer_ssl.py).

`PanelSample` deliberately exposes the same `feature_end_date`/`label_date`
attributes as dataset.py's `Sample`, so eval/walkforward.py's
generate_yearly_folds/split_masks/assert_no_leakage work completely
unchanged on this dataset (Python duck-typing -- they only ever touch those
two attributes on `.samples`, never the type itself).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class PanelSample:
    feature_end_date: dt.date
    label_date: dt.date
    entries: list[tuple[str, int]]  # (isin, array_idx) for every valid symbol on this date


class PanelSequenceDataset(Dataset):
    def __init__(
        self,
        features_df: pl.DataFrame,
        universe: list[str],
        seq_len: int = 120,
        min_symbols_per_date: int = 5,
        feature_columns: list[str] | None = None,
    ):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.seq_len = seq_len
        self.universe = list(universe)
        self._arrays: dict[str, dict] = {}

        df = features_df.filter(pl.col("isin").is_in(self.universe)).sort(["isin", "date"])
        for isin, grp in df.group_by("isin", maintain_order=True):
            isin_val = isin[0] if isinstance(isin, tuple) else isin
            feat = grp.select(self.feature_columns).to_numpy().astype(np.float32)
            target = grp[TARGET_COLUMN].to_numpy().astype(np.float32)
            dates = grp["date"].to_list()
            has_hist = grp["has_sufficient_history"].to_numpy()
            self._arrays[isin_val] = {
                "feat": feat, "target": target, "dates": dates, "has_hist": has_hist,
                "date_to_idx": {d: i for i, d in enumerate(dates)},
            }

        # Master calendar: sorted union of every date any universe member traded.
        all_dates = sorted({d for arr in self._arrays.values() for d in arr["dates"]})

        self.samples: list[PanelSample] = []
        for i in range(len(all_dates) - 1):  # need a label date, so stop one short
            t, label_date = all_dates[i], all_dates[i + 1]
            entries = []
            for isin, arr in self._arrays.items():
                idx = arr["date_to_idx"].get(t)
                if idx is None or idx < seq_len - 1:
                    continue
                if not arr["has_hist"][idx]:
                    continue
                window = arr["feat"][idx - seq_len + 1: idx + 1]
                y = arr["target"][idx]
                if np.isnan(window).any() or np.isnan(y):
                    continue
                entries.append((isin, idx))
            if len(entries) >= min_symbols_per_date:
                self.samples.append(PanelSample(feature_end_date=t, label_date=label_date, entries=entries))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n = len(s.entries)
        X = np.zeros((n, self.seq_len, len(self.feature_columns)), dtype=np.float32)
        y = np.zeros((n,), dtype=np.float32)
        for j, (isin, array_idx) in enumerate(s.entries):
            arr = self._arrays[isin]
            X[j] = arr["feat"][array_idx - self.seq_len + 1: array_idx + 1]
            y[j] = arr["target"][array_idx]
        return torch.from_numpy(X), torch.from_numpy(y)

    def isins_for(self, idx) -> list[str]:
        """The symbol identity for each row of __getitem__(idx)'s panel, in order --
        needed by the eval loop to attribute predictions back to a symbol/date pair."""
        return [isin for isin, _ in self.samples[idx].entries]

    def subset_by_mask(self, mask: np.ndarray) -> "PanelSequenceDataset":
        new = PanelSequenceDataset.__new__(PanelSequenceDataset)
        new.feature_columns = self.feature_columns
        new.seq_len = self.seq_len
        new.universe = self.universe
        new._arrays = self._arrays  # shared, read-only
        new.samples = [s for s, m in zip(self.samples, mask) if m]
        return new
