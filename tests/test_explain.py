"""Tests unitaires du module src.explain."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.explain import (  # noqa: E402
    compute_shap,
    group_share,
    mean_abs_shap,
    plot_contributions,
    transform_features,
)
from src.features import FEATURE_SETS  # noqa: E402
from src.models import make_model, positive_weight  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    """Petit modèle LightGBM pondéré, entraîné sur des fraudes « compte vidé »."""
    rng = np.random.default_rng(0)
    n = 1000
    old = rng.uniform(1, 1e5, n)
    fraud = rng.random(n) < 0.1
    amount = np.where(fraud, old, rng.uniform(1, 1e5, n))
    raw = pd.DataFrame(
        {
            "step": rng.integers(0, 240, n),
            "type": pd.Categorical(rng.choice(["TRANSFER", "CASH_OUT"], n)),
            "amount": amount,
            "oldbalanceOrig": old,
            "newbalanceOrig": np.where(fraud, 0.0, np.maximum(old - amount, 0)),
            "oldbalanceDest": rng.uniform(0, 1e5, n),
            "newbalanceDest": rng.uniform(0, 1e5, n),
        }
    )
    y = fraud.astype(int)
    model = make_model("lightgbm", "complet", "ponderation", pos_weight=positive_weight(y),
                       n_estimators=30).fit(raw, y)
    return model, raw


def test_transform_features_matches_model_inputs(fitted):
    model, raw = fitted
    features = transform_features(model, raw)
    assert list(features.columns) == FEATURE_SETS["complet"]
    assert len(features) == len(raw)


def test_shap_values_are_additive(fitted):
    """Base + somme des contributions = score brut du modèle (propriété de SHAP)."""
    model, raw = fitted
    values, features, base = compute_shap(model, raw)
    assert values.shape == features.shape
    raw_score = model.named_steps["model"].predict(features, raw_score=True)
    np.testing.assert_allclose(base + values.sum(axis=1), raw_score, atol=1e-5)


def test_mean_abs_shap_and_group_share(fitted):
    model, raw = fitted
    values, _, _ = compute_shap(model, raw)
    importance = mean_abs_shap(values)
    assert importance.is_monotonic_decreasing
    weighted = mean_abs_shap(values, weights=np.ones(len(values)))
    pd.testing.assert_series_equal(importance, weighted)
    groups = {"a": list(values.columns[:5]), "b": list(values.columns[5:])}
    assert group_share(values, groups).sum() == pytest.approx(1.0)


def test_plot_contributions_runs(fitted):
    model, raw = fitted
    values, features, base = compute_shap(model, raw)
    fig, ax = plt.subplots()
    plot_contributions(ax, values.iloc[0], features.iloc[0], base, max_features=4, title="essai")
    assert len(ax.patches) == 5  # 4 variables + « autres variables »
    plt.close(fig)
