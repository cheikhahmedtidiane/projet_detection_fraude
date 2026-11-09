"""Tests unitaires du module src.data (sur un petit échantillon synthétique)."""

import pandas as pd
import pytest

from src.data import (
    EXPECTED_COLUMNS,
    balance_consistency,
    load_data,
    load_raw,
    missing_values_report,
    quality_report,
    validate_schema,
)

RAW_HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
)
RAW_ROWS = [
    # Paiement cohérent côté émetteur, marchand sans solde renseigné
    "1,PAYMENT,100.0,C1,1000.0,900.0,M1,0.0,0.0,0,0",
    # Transfert frauduleux qui vide le compte, cohérent des deux côtés
    "1,TRANSFER,500.0,C2,500.0,0.0,C3,0.0,500.0,1,0",
    # Retrait frauduleux incohérent côté destinataire
    "2,CASH_OUT,500.0,C3,500.0,0.0,C4,1000.0,1000.0,1,0",
    # Dépôt : l'émetteur est crédité, l'agent destinataire est débité
    "3,CASH_IN,200.0,C5,50.0,250.0,C6,300.0,100.0,0,0",
]


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("\n".join([RAW_HEADER, *RAW_ROWS]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def df(raw_csv):
    return load_raw(raw_csv)


# ---------------------------------------------------------------- chargement
def test_load_raw_renames_and_types(df):
    assert list(df.columns) == EXPECTED_COLUMNS
    assert "oldbalanceOrg" not in df.columns
    assert df["step"].dtype == "int16"
    assert df["isFraud"].dtype == "int8"
    assert isinstance(df["type"].dtype, pd.CategoricalDtype)
    assert df["amount"].dtype == "float64"


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "absent.csv")


def test_load_data_creates_and_uses_cache(raw_csv, tmp_path):
    cache = tmp_path / "cache.parquet"
    first = load_data(raw_path=raw_csv, cache_path=cache)
    assert cache.exists()

    # Le CSV n'est plus nécessaire une fois le cache créé
    raw_csv.unlink()
    second = load_data(raw_path=raw_csv, cache_path=cache)
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------- validation
def test_validate_schema_ok(df):
    validate_schema(df)  # ne doit pas lever d'exception


def test_validate_schema_missing_column(df):
    with pytest.raises(ValueError, match="manquantes"):
        validate_schema(df.drop(columns="amount"))


def test_validate_schema_non_binary_target(df):
    bad = df.copy()
    bad.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="binaire"):
        validate_schema(bad)


# ---------------------------------------------------------------- qualité
def test_missing_values_report_no_missing(df):
    report = missing_values_report(df)
    assert report["n_missing"].sum() == 0
    assert set(report.index) == set(EXPECTED_COLUMNS)


def test_balance_consistency(df):
    result = balance_consistency(df)
    # Émetteur cohérent pour tous les types (y compris CASH_IN crédité)
    assert (result["pct_orig_coherent"] == 100).all()
    # CASH_IN : l'agent destinataire est débité (300 - 200 = 100)
    assert result.loc["CASH_IN", "pct_dest_coherent"] == 100
    assert result.loc["TRANSFER", "pct_dest_coherent"] == 100
    assert result.loc["CASH_OUT", "pct_dest_coherent"] == 0
    assert result.loc["PAYMENT", "pct_dest_coherent"] == 0


def test_quality_report(df):
    report = quality_report(df)
    assert report["n_rows"] == 4
    assert report["n_fraud"] == 2
    assert report["fraud_rate_pct"] == 50.0
    assert report["n_duplicated_rows"] == 0
    assert report["dest_prefixes"] == {"C": 3, "M": 1}
