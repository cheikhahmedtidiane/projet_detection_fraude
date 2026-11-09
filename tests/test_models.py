"""Tests unitaires du module src.models et du bootstrap de src.evaluate."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.evaluate import bootstrap_pr_auc, compute_metrics
from src.models import (
    IMBALANCE_STRATEGIES,
    MODEL_NAMES,
    make_classifier,
    make_model,
    make_sampler,
    positive_weight,
    temporal_cv_folds,
)


@pytest.fixture
def raw():
    """Petit jeu brut où les fraudes vident le compte et ont lieu la nuit."""
    rng = np.random.default_rng(0)
    n = 1500
    old = rng.uniform(1, 1e5, n)
    fraud = rng.random(n) < 0.1
    hour = np.where(fraud, rng.integers(0, 6, n), rng.integers(8, 22, n))
    amount = np.where(fraud, old, rng.uniform(1, 1e5, n))
    return pd.DataFrame(
        {
            "step": rng.integers(0, 10, n) * 24 + hour,
            "type": pd.Categorical(rng.choice(["TRANSFER", "CASH_OUT"], n)),
            "amount": amount,
            "oldbalanceOrig": old,
            "newbalanceOrig": np.where(fraud, 0.0, np.maximum(old - amount, 0)),
            "oldbalanceDest": rng.uniform(0, 1e5, n),
            "newbalanceDest": rng.uniform(0, 1e5, n),
            "isFraud": fraud.astype(int),
        }
    )


def test_positive_weight():
    assert positive_weight(np.array([0, 0, 0, 1])) == 3.0
    with pytest.raises(ValueError):
        positive_weight(np.zeros(5))


def test_make_classifier_applies_weight():
    rf = make_classifier("random_forest", pos_weight=50)
    assert rf.class_weight == {0: 1.0, 1: 50}
    assert make_classifier("random_forest").class_weight is None
    assert make_classifier("xgboost", pos_weight=50).get_params()["scale_pos_weight"] == 50
    lgbm = make_classifier("lightgbm", pos_weight=50, n_estimators=10)
    assert lgbm.get_params()["n_estimators"] == 10
    # Correctif de stabilité : pas d'initialisation à la prévalence brute si pondération
    assert lgbm.get_params()["boost_from_average"] is False
    assert make_classifier("lightgbm").get_params()["boost_from_average"] is True
    with pytest.raises(ValueError):
        make_classifier("svm")


def test_make_sampler():
    assert isinstance(make_sampler("sous_echantillonnage"), RandomUnderSampler)
    assert isinstance(make_sampler("smote"), SMOTE)
    assert make_sampler("aucune") is None and make_sampler("ponderation") is None
    with pytest.raises(ValueError):
        make_sampler("adasyn")


def test_make_model_requires_weight_for_ponderation():
    with pytest.raises(ValueError):
        make_model("xgboost", imbalance="ponderation")


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("imbalance", IMBALANCE_STRATEGIES)
def test_all_models_learn(raw, name, imbalance):
    y = raw["isFraud"]
    model = make_model(
        name, "complet", imbalance, pos_weight=positive_weight(y), sampling_ratio=0.5, n_estimators=30
    ).fit(raw, y)
    proba = model.predict_proba(raw)[:, 1]
    assert proba.shape == (len(raw),)
    assert compute_metrics(y, proba)["pr_auc"] > 0.9


def test_sampler_only_applied_at_fit(raw):
    """Le rééchantillonnage ne doit pas modifier le nombre de prédictions."""
    model = make_model(
        "lightgbm", "transaction", "sous_echantillonnage", sampling_ratio=0.5, n_estimators=10
    )
    model.fit(raw, raw["isFraud"])
    assert len(model.predict_proba(raw)) == len(raw)


def test_temporal_cv_folds_never_use_future():
    steps = np.repeat(np.arange(13 * 24), 3)  # 13 jours
    folds = temporal_cv_folds(steps, n_folds=3, val_days=2)
    assert len(folds) == 3
    days = steps // 24
    expected_val_days = [{7, 8}, {9, 10}, {11, 12}]
    for (tr, va), val_days in zip(folds, expected_val_days):
        assert days[tr].max() < days[va].min()
        assert set(days[va]) == val_days
        assert days[tr].min() == 0
    with pytest.raises(ValueError):
        temporal_cv_folds(steps, n_folds=7, val_days=2)


def test_bootstrap_pr_auc_paired():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    good = y + rng.normal(0, 0.3, len(y))
    bad = rng.random(len(y))
    table = bootstrap_pr_auc(y, {"bon": good, "hasard": bad}, reference="hasard", n_boot=50)
    assert table.loc["bon", "ic_bas"] <= table.loc["bon", "pr_auc"] <= table.loc["bon", "ic_haut"]
    assert table.loc["bon", "ecart_ic_bas"] > 0
    assert table.loc["bon", "p_meilleur"] == 1.0
    assert table.loc["hasard", "ecart_ref"] == 0
