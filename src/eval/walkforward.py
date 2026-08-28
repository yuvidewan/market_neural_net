"""
Purged, embargoed, expanding-window walk-forward splitting (README §4.1).

No shuffled k-fold anywhere in this repo -- splits are always chronological,
and a training sample is dropped ("purged") if its label only becomes known
too close to (or inside) the test window, per López de Prado. Embargo
calendar days is a conservative proxy for embargo trading-sessions (it purges
at least as much as a trading-day embargo would, never less).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from src.data.dataset import SequenceDataset


@dataclass
class Fold:
    label: str
    test_start: dt.date
    test_end: dt.date


def generate_yearly_folds(years: list[int]) -> list[Fold]:
    """One fold per calendar year in `years` (each becomes a test year in an
    expanding-window scheme; training is implicitly "everything before it")."""
    return [
        Fold(label=str(y), test_start=dt.date(y, 1, 1), test_end=dt.date(y, 12, 31))
        for y in years
    ]


def split_masks(
    dataset: SequenceDataset,
    fold: Fold,
    embargo_days: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    train_mask: samples whose label_date resolves strictly before
      (fold.test_start - embargo_days) -- i.e. everything historical, purged
      of any sample bleeding into the embargo window.
    test_mask: samples whose feature window ends on/after test_start AND
      whose label resolves on/before test_end -- strictly inside the fold.
    A sample can be in neither mask (it's in the embargo gap, or entirely
    after the fold) -- that's intentional, not a bug.
    """
    embargo_cutoff = fold.test_start - dt.timedelta(days=embargo_days)
    train_mask = np.empty(len(dataset.samples), dtype=bool)
    test_mask = np.empty(len(dataset.samples), dtype=bool)
    for i, s in enumerate(dataset.samples):
        train_mask[i] = s.label_date < embargo_cutoff
        test_mask[i] = (fold.test_start <= s.feature_end_date) and (s.label_date <= fold.test_end)
    return train_mask, test_mask


def assert_no_leakage(train_dataset: SequenceDataset, fold: Fold, embargo_days: int = 10) -> None:
    """
    Independent runtime guard, NOT a re-derivation of split_masks: call this
    on the actual dataset object a training script is about to hand to the
    DataLoader (i.e. after `ds.subset_by_mask(train_mask)`), right before
    training starts. It re-checks every sample directly against the embargo
    rule, so it catches bugs in how a caller assembled that subset (wrong
    mask, an off-by-one, a manually-added sample) -- not just bugs in
    split_masks itself, which this function never calls.
    """
    embargo_cutoff = fold.test_start - dt.timedelta(days=embargo_days)
    for s in train_dataset.samples:
        if s.label_date >= embargo_cutoff:
            raise AssertionError(
                f"leakage: train sample label_date={s.label_date} >= embargo_cutoff={embargo_cutoff}"
            )
