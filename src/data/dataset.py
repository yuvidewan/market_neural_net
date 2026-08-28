"""
Causal sequence windowing + a torch Dataset over the features built in
features.py. A sample is: the trailing `seq_len` daily feature-bars for one
ISIN ending at day t (the model's input), and the realized next-day return
(the label). Windows never cross an ISIN boundary.

Two dates matter for every sample, and the walk-forward harness (eval/walkforward.py)
uses both:
  - `feature_end_date`: the last day of data the model actually SEES (day t).
  - `label_date`: the day whose return is being predicted (day t+1) -- this is
    the date that must be purged/embargoed against, not feature_end_date,
    because the sample's true label only becomes KNOWN on label_date.
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
class Sample:
    isin: str
    feature_end_date: dt.date
    label_date: dt.date
    array_idx: int  # position within that isin's array, i.e. the window's last row


class SequenceDataset(Dataset):
    def __init__(
        self,
        features_df: pl.DataFrame,
        seq_len: int = 120,
        feature_columns: list[str] | None = None,
        shuffle_labels: bool = False,
        shuffle_seed: int = 0,
    ):
        """
        shuffle_labels: if True, permutes the target values across the
        dataset's samples AFTER building the valid-window index (so the same
        set of windows is used either way) -- this is the anti-fooling
        control: with labels shuffled, X no longer has any real relationship
        to y, so a model that still finds tradeable signal indicates a leak
        somewhere in the pipeline, not real predictive power.
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.seq_len = seq_len
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self.samples: list[Sample] = []

        df = features_df.sort(["isin", "date"])
        for isin, grp in df.group_by("isin", maintain_order=True):
            isin_val = isin[0] if isinstance(isin, tuple) else isin
            feat = grp.select(self.feature_columns).to_numpy().astype(np.float32)
            target = grp[TARGET_COLUMN].to_numpy().astype(np.float32)
            dates = grp["date"].to_list()
            has_hist = grp["has_sufficient_history"].to_numpy()

            self._arrays[isin_val] = {"feat": feat, "target": target, "dates": dates}

            n = feat.shape[0]
            for end_row in range(seq_len - 1, n - 1):  # -1: need a label at end_row+1
                if not has_hist[end_row]:
                    continue
                window = feat[end_row - seq_len + 1: end_row + 1]
                y = target[end_row]
                if np.isnan(window).any() or np.isnan(y):
                    continue
                self.samples.append(Sample(
                    isin=isin_val,
                    feature_end_date=dates[end_row],
                    label_date=dates[end_row + 1],
                    array_idx=end_row,
                ))

        if shuffle_labels:
            rng = np.random.default_rng(shuffle_seed)
            order = rng.permutation(len(self.samples))
            true_targets = [self._arrays[s.isin]["target"][s.array_idx] for s in self.samples]
            shuffled_targets = [true_targets[i] for i in order]
            self._shuffled_target = shuffled_targets
        else:
            self._shuffled_target = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        arr = self._arrays[s.isin]
        window = arr["feat"][s.array_idx - self.seq_len + 1: s.array_idx + 1]
        if self._shuffled_target is not None:
            y = self._shuffled_target[idx]
        else:
            y = arr["target"][s.array_idx]
        return torch.from_numpy(window), torch.tensor(y, dtype=torch.float32)

    def subset_by_mask(self, mask: np.ndarray) -> "SequenceDataset":
        """Return a shallow view containing only samples[i] where mask[i] is True."""
        new = SequenceDataset.__new__(SequenceDataset)
        new.feature_columns = self.feature_columns
        new.seq_len = self.seq_len
        new._arrays = self._arrays  # shared, read-only
        new.samples = [s for s, m in zip(self.samples, mask) if m]
        if self._shuffled_target is not None:
            new._shuffled_target = [t for t, m in zip(self._shuffled_target, mask) if m]
        else:
            new._shuffled_target = None
        return new
