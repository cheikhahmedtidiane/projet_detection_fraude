"""Tests unitaires du module src.features."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_SETS,
    BALANCE_FEATURES,
    TRANSACTION_FEATURES,
    ColumnSelector,
    FraudFeatureBuilder,
    build_preprocessor,
    signed_log1p,
)


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            # 0h, 5h, 12h, 23h
            "step": [24, 5, 12, 23],
            "type": pd.Categorical(["TRANSFER", "CASH_OUT", "CASH_OUT", "TRANSFER"]),
            "amount": [1000.0, 500.0, 200.0, 300.0],
            "oldbalanceOrig": [1000.0, 0.0, 1000.0, 300.0],
            "newbalanceOrig": [0.0, 0.0, 800.0, 0.0],
            "oldbalanceDest": [0.0, 100.0, 50.0, 0.0],
            "newbalanceDest": [0.0, 600.0, 250.0, 300.0],
            "isFraud": [1, 0, 0, 1],
        }
    )


@pytest.fixture
def feats(raw):
    return FraudFeatureBuilder().fit_transform(raw)


def test_feature_sets_are_consistent():
    assert FEATURE_SETS["complet"] == TRANSACTION_FEATURES + BALANCE_FEATURES
    assert not set(FEATURE_SETS["transaction"]) & set(BALANCE_FEATURES)


def test_builder_outputs_all_features_without_nan(feats):
    assert list(feats.columns) == FEATURE_SETS["complet"]
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy(dtype=float)).all()


def test_builder_does_not_use_target(raw):
    a = FraudFeatureBuilder().fit_transform(raw)
    b = FraudFeatureBuilder().fit_transform(raw.drop(columns="isFraud"))
    pd.testing.assert_frame_equal(a, b)


def test_transaction_features(feats):
    assert feats["is_transfer"].tolist() == [1, 0, 0, 1]
    assert feats["is_night"].tolist() == [1, 1, 0, 0]
    # 0h et 23h sont proches sur le cercle horaire
    h0 = feats.loc[0, ["heure_sin", "heure_cos"]].to_numpy()
    h23 = feats.loc[3, ["heure_sin", "heure_cos"]].to_numpy()
    h12 = feats.loc[2, ["heure_sin", "heure_cos"]].to_numpy()
    assert np.linalg.norm(h0 - h23) < np.linalg.norm(h0 - h12)


def test_balance_flags(feats):
    assert feats["compte_vide"].tolist() == [1, 0, 0, 1]
    assert feats["orig_solde_nul"].tolist() == [0, 1, 0, 0]
    assert feats["dest_soldes_nuls"].tolist() == [1, 0, 0, 0]


def test_balance_errors(feats):
    # ligne 2 : cohérente des deux côtés -> erreurs nulles
    assert feats.loc[2, "erreur_orig"] == pytest.approx(0.0)
    assert feats.loc[2, "erreur_dest"] == pytest.approx(0.0)
    # ligne 1 : solde émetteur insuffisant -> erreur émetteur positive
    assert feats.loc[1, "erreur_orig"] == pytest.approx(np.log1p(500.0))
    # ligne 0 : destinataire non crédité -> erreur destinataire positive
    assert feats.loc[0, "erreur_dest"] == pytest.approx(np.log1p(1000.0))


def test_signed_log1p_keeps_sign():
    out = signed_log1p(pd.Series([-99.0, 0.0, 99.0]))
    assert out.tolist() == pytest.approx([-np.log(100), 0.0, np.log(100)])


def test_builder_missing_column(raw):
    with pytest.raises(ValueError, match="manquantes"):
        FraudFeatureBuilder().fit(raw.drop(columns="amount"))


def test_column_selector_order(feats):
    out = ColumnSelector(["log_amount", "is_transfer"]).fit_transform(feats)
    assert list(out.columns) == ["log_amount", "is_transfer"]


@pytest.mark.parametrize("feature_set", ["complet", "transaction"])
def test_build_preprocessor(raw, feature_set):
    out = build_preprocessor(feature_set).fit_transform(raw)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == FEATURE_SETS[feature_set]


@pytest.mark.parametrize("scale", [False, True])
def test_build_preprocessor_fit_then_transform(raw, scale):
    # Cas d'usage réel : fit sur le train, puis transform séparé (sur d'autres données)
    pre = build_preprocessor("complet", scale=scale).fit(raw)
    out = pre.transform(raw.iloc[:2])
    assert out.shape == (2, len(FEATURE_SETS["complet"]))


def test_build_preprocessor_scaling_is_fitted_on_train_only(raw):
    pre = build_preprocessor("transaction", scale=True).fit(raw.iloc[:2])
    train_out = pre.transform(raw.iloc[:2])
    assert train_out["log_amount"].mean() == pytest.approx(0.0, abs=1e-9)
    # Sur d'autres données, la moyenne n'est plus nulle : le scaler n'a pas été réappris
    other_out = pre.transform(raw.iloc[2:])
    assert other_out["log_amount"].mean() != pytest.approx(0.0, abs=1e-3)


def test_build_preprocessor_unknown_set():
    with pytest.raises(ValueError):
        build_preprocessor("inconnu")
