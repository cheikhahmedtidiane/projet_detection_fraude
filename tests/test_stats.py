"""Tests unitaires du module src.stats."""

import numpy as np
import pandas as pd
import pytest

from src.stats import (
    chi2_independence,
    interpret_cramers_v,
    interpret_rank_biserial,
    mann_whitney,
    odds_ratio,
    results_table,
)


def test_chi2_perfect_association():
    x = pd.Series(["a"] * 50 + ["b"] * 50)
    y = pd.Series([0] * 50 + [1] * 50)
    res = chi2_independence(x, y)
    assert res.effect_size == pytest.approx(1.0)
    assert res.p_value < 1e-10
    assert res.effect_label == "fort"


def test_chi2_independence_no_association():
    x = pd.Series(["a", "b"] * 50)
    y = pd.Series([0, 0, 1, 1] * 25)
    res = chi2_independence(x, y)
    assert res.effect_size == pytest.approx(0.0)
    assert res.effect_label == "négligeable"


def test_mann_whitney_full_separation():
    res = mann_whitney(pd.Series([10, 11, 12]), pd.Series([1, 2, 3]))
    assert res.effect_size == pytest.approx(1.0)
    res_inv = mann_whitney(pd.Series([1, 2, 3]), pd.Series([10, 11, 12]))
    assert res_inv.effect_size == pytest.approx(-1.0)


def test_mann_whitney_identical_distributions():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=2000))
    b = pd.Series(rng.normal(size=2000))
    res = mann_whitney(a, b)
    assert abs(res.effect_size) < 0.1


def test_odds_ratio():
    # exposés : 9 issues / 1 non ; non exposés : 1 issue / 9 non  ->  OR ≈ 81 (Haldane)
    exposure = pd.Series([True] * 10 + [False] * 10)
    outcome = pd.Series([1] * 9 + [0] + [1] + [0] * 9)
    assert odds_ratio(exposure, outcome) == pytest.approx((9.5 * 9.5) / (1.5 * 1.5))


@pytest.mark.parametrize(
    "v, expected", [(0.05, "négligeable"), (0.2, "faible"), (0.4, "moyen"), (0.6, "fort")]
)
def test_interpret_cramers_v(v, expected):
    assert interpret_cramers_v(v, dof=1) == expected


def test_interpret_rank_biserial_uses_absolute_value():
    assert interpret_rank_biserial(-0.6) == "fort"


def test_results_table_bonferroni():
    r1 = chi2_independence(pd.Series(["a"] * 50 + ["b"] * 50), pd.Series([0] * 50 + [1] * 50))
    r2 = chi2_independence(pd.Series(["a", "b"] * 50), pd.Series([0, 0, 1, 1] * 25))
    table = results_table({"H1": r1, "H2": r2})
    assert table.attrs["alpha_bonferroni"] == pytest.approx(0.025)
    assert table.loc["H1", "significatif"]
    assert not table.loc["H2", "significatif"]
