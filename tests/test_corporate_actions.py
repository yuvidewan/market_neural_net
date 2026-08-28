"""
Ratio-parser tests against real subject strings sampled from NSE's own
corporate-actions API (2015-2021), not synthetic examples.
"""
import math

import pytest

from src.data.corporate_actions import parse_corporate_action_ratio


def approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=tol)


@pytest.mark.parametrize("subject,expected", [
    ("Bonus 1:1", 1 / 2),
    ("Bonus 1 : 1", 1 / 2),
    ("Bonus 1: 2", 2 / 3),
    ("Bonus 10:1", 1 / 11),
    ("Bonus 2:1", 1 / 3),
    ("Bonus 1 : 1250", 1250 / 1251),
    ("Annual General Meeting / Dividend - Rs 3/- Per Share / Bonus - 1:2", 2 / 3),
])
def test_bonus_only(subject, expected):
    got = parse_corporate_action_ratio(subject)
    assert got is not None
    assert approx(got, expected)


@pytest.mark.parametrize("subject,expected", [
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 1 / 10),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", 2 / 10),
    ("Face Value Split (Sub-Division) - From Rs 10 Per Share To Rs 5  Per Share", 5 / 10),
    ("Face Value Split From Rs 10 To Re 1", 1 / 10),
    ("Face Value Split From Rs 2 To Re 1", 1 / 2),
    ("Face Valus Split (Sub-Division) - From Rs 10/- Per To Rs 2/- Per Share", 2 / 10),  # real typo "Valus"
])
def test_split_only(subject, expected):
    got = parse_corporate_action_ratio(subject)
    assert got is not None
    assert approx(got, expected)


@pytest.mark.parametrize("subject,expected", [
    ("Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", 0.5 * 0.2),
    ("Bonus 1:1 / Face Value Split - From Rs 10/- Per Share To Rs 5/- Per Share", 0.5 * 0.5),
    ("Bonus 4:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 5/- Per Share", (1 / 5) * 0.5),
    (
        "Interim Dividend - Rs 1.50/- Per Share / Face Value Split - From Rs 10/- Per Share "
        "To Rs 5/- Per Share (Purpose Revised)",
        0.5,  # dividend clause present but must not block the split parse
    ),
])
def test_combined_bonus_and_split(subject, expected):
    got = parse_corporate_action_ratio(subject)
    assert got is not None
    assert approx(got, expected)


@pytest.mark.parametrize("subject", [
    "Scheme Of Arrangement - Bonus Debentures 1:1",   # debenture, not equity share bonus
    "Scheme Of Arangement- Bonus - 1 Debenture For 1 Equity Share Held",
    "Demerger",
    "Merger/Demerger",
    "Scheme Of Demerger",
    "Scheme Of Arrangement In The Nature Of Demerger",
])
def test_excluded_actions_return_none(subject):
    assert parse_corporate_action_ratio(subject) is None


@pytest.mark.parametrize("subject", [
    "Interim Dividend - Rs 6 Per Share",     # plain dividend: no ratio to extract, correctly None
    "Annual General Meeting",
    "",
])
def test_non_actions_return_none(subject):
    assert parse_corporate_action_ratio(subject) is None
