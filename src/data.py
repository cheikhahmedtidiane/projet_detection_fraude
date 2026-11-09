"""Chargement, validation et contrôle qualité des données de transactions.

Ce module est le point d'entrée unique vers les données : tous les notebooks
et scripts passent par :func:`load_data` afin de garantir des types et des
noms de colonnes identiques partout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CLEAN_PARQUET_PATH, RAW_CSV_PATH, TRANSACTION_TYPES

# ---------------------------------------------------------------------------
# Schéma
# ---------------------------------------------------------------------------
# Les montants et soldes restent en float64 : en float32, la précision
# (~7 chiffres significatifs) produirait des erreurs d'arrondi de l'ordre du
# centime sur des montants de plusieurs millions, ce qui fausserait les
# futures variables d'« erreur de solde ».
RAW_DTYPES: dict[str, str] = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# Harmonisation : la colonne source « oldbalanceOrg » est incohérente avec
# « newbalanceOrig ». On la renomme pour un nommage uniforme.
COLUMN_RENAMES: dict[str, str] = {"oldbalanceOrg": "oldbalanceOrig"}

EXPECTED_COLUMNS: list[str] = [COLUMN_RENAMES.get(c, c) for c in RAW_DTYPES]

BALANCE_COLUMNS: list[str] = [
    "oldbalanceOrig",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_raw(path: str | Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Charge le CSV brut avec des types optimisés et des colonnes harmonisées.

    Args:
        path: chemin du fichier CSV brut.

    Returns:
        DataFrame typé et validé.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si le schéma ne correspond pas à celui attendu.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Téléchargez le dataset depuis Kaggle et placez-le dans data/raw/ "
            "(voir README)."
        )

    df = pd.read_csv(path, dtype=RAW_DTYPES)
    df = df.rename(columns=COLUMN_RENAMES)
    validate_schema(df)
    return df


def load_data(
    use_cache: bool = True,
    raw_path: str | Path = RAW_CSV_PATH,
    cache_path: str | Path = CLEAN_PARQUET_PATH,
) -> pd.DataFrame:
    """Charge les données en privilégiant le cache Parquet.

    Le premier appel lit le CSV (lent, ~470 Mo) puis écrit une copie Parquet ;
    les appels suivants relisent directement le Parquet (beaucoup plus rapide
    et plus léger sur disque).

    Args:
        use_cache: si True, lit/écrit le cache Parquet.
        raw_path: chemin du CSV brut.
        cache_path: chemin du fichier Parquet de cache.

    Returns:
        DataFrame typé et validé.
    """
    cache_path = Path(cache_path)
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        validate_schema(df)
        return df

    df = load_raw(raw_path)
    if use_cache:
        save_parquet(df, cache_path)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path = CLEAN_PARQUET_PATH) -> Path:
    """Sauvegarde un DataFrame au format Parquet (crée le dossier si besoin)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Vérifie que le DataFrame respecte le schéma attendu.

    Contrôles : présence et ordre des colonnes, cible binaire, types de
    transaction connus.

    Raises:
        ValueError: au premier écart détecté.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        raise ValueError(f"Colonnes inattendues : {sorted(extra)}")

    for col in ("isFraud", "isFlaggedFraud"):
        values = set(df[col].dropna().unique())
        if not values <= {0, 1}:
            raise ValueError(f"La colonne {col} doit être binaire, trouvé : {values}")

    unknown_types = set(df["type"].dropna().unique()) - set(TRANSACTION_TYPES)
    if unknown_types:
        raise ValueError(f"Types de transaction inconnus : {unknown_types}")


# ---------------------------------------------------------------------------
# Contrôle qualité
# ---------------------------------------------------------------------------
def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et pourcentage de valeurs manquantes par colonne."""
    n_missing = df.isna().sum()
    return pd.DataFrame(
        {"n_missing": n_missing, "pct_missing": (n_missing / len(df) * 100).round(4)}
    )


def balance_consistency(df: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """Mesure la cohérence comptable des soldes, par type de transaction.

    Pour une transaction cohérente, on attend :
      * émetteur     : oldbalanceOrig - amount ≈ newbalanceOrig
      * destinataire : oldbalanceDest + amount ≈ newbalanceDest

    Pour CASH_IN (dépôt d'espèces auprès d'un agent), le sens est inversé :
    l'émetteur est crédité et le destinataire (l'agent) est débité.

    Args:
        df: données de transactions.
        tol: tolérance absolue (en unités monétaires) pour l'égalité.

    Returns:
        Pourcentage de transactions cohérentes côté émetteur et destinataire,
        par type de transaction.
    """
    # +1 pour CASH_IN (sens inversé), -1 pour les autres types
    sign = df["type"].map({"CASH_IN": 1.0}).astype("float64").fillna(-1.0)
    orig_ok = (df["oldbalanceOrig"] + sign * df["amount"] - df["newbalanceOrig"]).abs() <= tol
    dest_ok = (df["oldbalanceDest"] - sign * df["amount"] - df["newbalanceDest"]).abs() <= tol

    result = (
        pd.DataFrame({"type": df["type"], "orig_coherent": orig_ok, "dest_coherent": dest_ok})
        .groupby("type", observed=True)
        .mean()
        .mul(100)
        .round(2)
    )
    return result.rename(
        columns={"orig_coherent": "pct_orig_coherent", "dest_coherent": "pct_dest_coherent"}
    )


def quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Synthèse des principaux indicateurs de qualité des données.

    Returns:
        Dictionnaire d'indicateurs (volumétrie, doublons, valeurs aberrantes
        évidentes, cohérence des identifiants, déséquilibre de la cible).
    """
    n = len(df)
    dest_prefix = df["nameDest"].str[0]
    orig_prefix = df["nameOrig"].str[0]

    return {
        "n_rows": n,
        "n_columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
        "n_missing_total": int(df.isna().sum().sum()),
        "n_duplicated_rows": int(df.duplicated().sum()),
        "n_negative_amount": int((df["amount"] < 0).sum()),
        "n_zero_amount": int((df["amount"] == 0).sum()),
        "n_negative_balance": int((df[BALANCE_COLUMNS] < 0).any(axis=1).sum()),
        "step_min": int(df["step"].min()),
        "step_max": int(df["step"].max()),
        "n_unique_orig": int(df["nameOrig"].nunique()),
        "n_unique_dest": int(df["nameDest"].nunique()),
        "orig_prefixes": orig_prefix.value_counts().to_dict(),
        "dest_prefixes": dest_prefix.value_counts().to_dict(),
        "n_fraud": int(df["isFraud"].sum()),
        "fraud_rate_pct": round(df["isFraud"].mean() * 100, 4),
        "n_flagged": int(df["isFlaggedFraud"].sum()),
    }
