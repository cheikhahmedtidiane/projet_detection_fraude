"""Tests du pipeline d'entraînement (src.train) sur un petit jeu synthétique."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS
from src.train import MODEL_SPECS, load_model, parse_args, run_training


@pytest.fixture(scope="module")
def raw():
    """Jeu brut au schéma Kaggle, couvrant les jours 0 à 20 (train, val, test, drift)."""
    rng = np.random.default_rng(0)
    n = 6000
    old = rng.uniform(1, 1e5, n)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n, p=[0.4, 0.4, 0.2])
    fraud = (rng.random(n) < 0.05) & (types != "PAYMENT")
    amount = np.where(fraud, old, rng.uniform(1, 1e5, n))
    df = pd.DataFrame(
        {
            "step": np.sort(rng.integers(0, 21 * 24, n)),
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrig": old,
            "newbalanceOrig": np.where(fraud, 0.0, np.maximum(old - amount, 0)),
            "nameDest": [f"C{i + n}" for i in range(n)],
            "oldbalanceDest": rng.uniform(0, 1e5, n),
            "newbalanceDest": rng.uniform(0, 1e5, n),
            "isFraud": fraud.astype(int),
            "isFlaggedFraud": 0,
        }
    )
    return df[EXPECTED_COLUMNS]


def test_run_training_end_to_end(raw, tmp_path):
    results = run_training(raw, ["lightgbm_complet"], cost_fp=100, output_dir=tmp_path)
    meta = results["lightgbm_complet"]
    # les fraudes vident le compte : le jeu complet est (presque) parfaitement séparable
    assert meta["validation"]["pr_auc"] > 0.95
    assert 0 < meta["seuil"] <= 1
    assert meta["entrainement"]["jours"][1] <= 12  # le train s'arrête au jour 12
    assert meta["cout_fausse_alerte"] == 100

    pipeline, saved = load_model("lightgbm_complet", tmp_path)
    assert saved == json.loads(json.dumps(meta))  # JSON relu identique
    val_like = raw[raw["type"].isin(["TRANSFER", "CASH_OUT"])].head(50)
    assert pipeline.predict_proba(val_like).shape == (50, 2)


def test_run_training_without_saving(raw, tmp_path):
    results = run_training(raw, ["lightgbm_transaction"], output_dir=None)
    assert set(results) == {"lightgbm_transaction"}
    assert results["lightgbm_transaction"]["variables"] == MODEL_SPECS["lightgbm_transaction"].params \
        and False or results["lightgbm_transaction"]["variables"][0] == "is_transfer"


def test_unknown_model_is_rejected(raw):
    with pytest.raises(ValueError):
        run_training(raw, ["svm"], output_dir=None)


def test_parse_args_defaults_and_options():
    args = parse_args([])
    assert args.models == list(MODEL_SPECS) and not args.no_cache
    args = parse_args(["--models", "lightgbm_transaction", "--cost-fp", "2000"])
    assert args.models == ["lightgbm_transaction"] and args.cost_fp == 2000
