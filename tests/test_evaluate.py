"""Tests unitaires des modules src.evaluate et src.baselines."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import RegleCompteVide, RegleIsFlagged, make_logistic_regression
from src.evaluate import business_cost, compare_models, compute_metrics, save_results


# ---------------------------------------------------------------- évaluation
def test_compute_metrics_perfect_model():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision"] == m["rappel"] == m["f2"] == pytest.approx(1.0)
    assert (m["vp"], m["fp"], m["fn"], m["vn"]) == (2, 0, 0, 2)


def test_compute_metrics_threshold_effect():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    m_low = compute_metrics(y, scores, threshold=0.3)
    m_high = compute_metrics(y, scores, threshold=0.7)
    assert m_low["rappel"] == 1.0 and m_low["n_alertes"] == 3
    assert m_high["rappel"] == 0.5 and m_high["precision"] == 1.0


def test_compute_metrics_no_alert_has_zero_precision():
    m = compute_metrics(np.array([0, 1]), np.array([0.1, 0.2]), threshold=0.5)
    assert m["precision"] == 0.0 and m["n_alertes"] == 0


def test_business_cost():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    amounts = np.array([1000.0, 3000.0, 50.0, 50.0])
    cost = business_cost(y_true, y_pred, amounts, cost_fp=100)
    assert cost["cout_fraudes_ratees"] == 3000.0
    assert cost["cout_fausses_alertes"] == 100.0
    assert cost["cout_total"] == 3100.0
    assert cost["montant_detecte_pct"] == pytest.approx(25.0)


def test_compare_models_and_save(tmp_path):
    y = np.array([0, 0, 1, 1])
    table = compare_models(
        {"a": np.array([0.1, 0.2, 0.8, 0.9]), "b": np.array([0.9, 0.8, 0.2, 0.1])},
        y,
        amounts=np.ones(4),
    )
    assert list(table.index) == ["a", "b"]
    assert table.loc["a", "pr_auc"] > table.loc["b", "pr_auc"]
    path = save_results(table, "essai", tmp_path)
    assert pd.read_csv(path, index_col="modele").shape[0] == 2


# ---------------------------------------------------------------- baselines
@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    n = 400
    old = rng.uniform(0, 1e5, n)
    fraud = rng.random(n) < 0.2
    amount = np.where(fraud, old, rng.uniform(1, 1e5, n))
    return pd.DataFrame(
        {
            "step": rng.integers(1, 300, n),
            "type": pd.Categorical(rng.choice(["TRANSFER", "CASH_OUT"], n)),
            "amount": amount,
            "oldbalanceOrig": old,
            "newbalanceOrig": np.where(fraud, 0.0, np.maximum(old - amount, 0)),
            "oldbalanceDest": rng.uniform(0, 1e5, n),
            "newbalanceDest": rng.uniform(0, 1e5, n),
            "isFlaggedFraud": (fraud & (amount > 9e4)).astype(int),
            "isFraud": fraud.astype(int),
        }
    )


def test_regle_compte_vide(raw):
    rule = RegleCompteVide().fit(raw)
    pred = rule.predict(raw)
    assert (pred == raw["isFraud"]).all()
    proba = rule.predict_proba(raw)
    assert proba.shape == (len(raw), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_regle_is_flagged(raw):
    pred = RegleIsFlagged().fit(raw).predict(raw)
    assert (pred == raw["isFlaggedFraud"]).all()


def test_logistic_regression_learns(raw):
    model = make_logistic_regression("complet").fit(raw, raw["isFraud"])
    proba = model.predict_proba(raw)[:, 1]
    assert compute_metrics(raw["isFraud"], proba)["roc_auc"] > 0.9
