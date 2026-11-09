"""Protocole expérimental : découpage temporel train / validation / test / drift.

Le découpage est **temporel** (le modèle apprend sur le passé et est évalué sur
le futur) et tient compte de la non-stationnarité mise en évidence en EDA :

* ``train``  : jours 0 à 12  → apprentissage ;
* ``val``    : jours 13 à 14 → réglage des hyperparamètres et du seuil ;
* ``test``   : jours 15 à 16 → évaluation finale, utilisée **une seule fois** ;
* ``drift``  : jours 17 à 30 → période où l'activité légitime du simulateur
  s'effondre ; sert de **test de robustesse** à un changement de distribution.

Voir ``notebooks/03_protocole_experimental.ipynb`` pour la justification.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    FRAUD_TYPES,
    SPLIT_NAMES,
    SPLITS_DIR,
    TARGET,
    TEST_LAST_DAY,
    TIME_COL,
    TRAIN_LAST_DAY,
    VAL_LAST_DAY,
)

HOURS_PER_DAY = 24


def step_to_day(step: pd.Series) -> pd.Series:
    """Convertit un step (heure de simulation) en jour de simulation."""
    return step // HOURS_PER_DAY


def filter_fraud_types(df: pd.DataFrame, types: list[str] = FRAUD_TYPES) -> pd.DataFrame:
    """Ne conserve que les types de transaction susceptibles d'être frauduleux.

    Les autres types (CASH_IN, PAYMENT, DEBIT) ne contiennent aucune fraude
    (hypothèse H1) : en production, ils seraient acceptés sans scoring.
    """
    out = df[df["type"].isin(types)].copy()
    if isinstance(out["type"].dtype, pd.CategoricalDtype):
        out["type"] = out["type"].cat.remove_unused_categories()
    return out


def assign_split(
    step: pd.Series,
    train_last_day: int = TRAIN_LAST_DAY,
    val_last_day: int = VAL_LAST_DAY,
    test_last_day: int = TEST_LAST_DAY,
) -> pd.Series:
    """Associe chaque transaction à un jeu de données selon son jour.

    Returns:
        Série catégorielle ordonnée à valeurs dans {train, val, test, drift}.
    """
    if not train_last_day < val_last_day < test_last_day:
        raise ValueError("Les bornes doivent vérifier train < val < test.")

    day = step_to_day(step)
    labels = pd.Series("drift", index=step.index, dtype="object")
    labels[day <= test_last_day] = "test"
    labels[day <= val_last_day] = "val"
    labels[day <= train_last_day] = "train"
    return labels.astype(pd.CategoricalDtype(SPLIT_NAMES, ordered=True))


def temporal_split(df: pd.DataFrame, **bounds: int) -> dict[str, pd.DataFrame]:
    """Découpe un DataFrame en jeux train / val / test / drift.

    Args:
        df: données contenant la colonne ``step``.
        **bounds: bornes optionnelles (train_last_day, val_last_day, test_last_day).

    Returns:
        Dictionnaire {nom du jeu: DataFrame}, dans l'ordre chronologique.
    """
    labels = assign_split(df[TIME_COL], **bounds)
    return {name: df[labels == name] for name in SPLIT_NAMES}


def check_no_temporal_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Vérifie que chaque jeu est strictement postérieur au précédent.

    Raises:
        ValueError: si un jeu contient une transaction antérieure ou simultanée
            à la dernière transaction du jeu précédent.
    """
    previous_name, previous_max = None, None
    for name in SPLIT_NAMES:
        part = splits.get(name)
        if part is None or part.empty:
            continue
        if previous_max is not None and part[TIME_COL].min() <= previous_max:
            raise ValueError(f"Chevauchement temporel entre {previous_name} et {name}.")
        previous_name, previous_max = name, part[TIME_COL].max()


def summarize_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tableau récapitulatif : volumes, fraudes, taux de fraude et période."""
    total = sum(len(part) for part in splits.values())
    rows = {}
    for name, part in splits.items():
        days = step_to_day(part[TIME_COL])
        rows[name] = {
            "jours": f"{days.min()}–{days.max()}" if len(part) else "-",
            "n_transactions": len(part),
            "pct_transactions": len(part) / total * 100 if total else 0.0,
            "n_fraudes": int(part[TARGET].sum()),
            "taux_fraude_pct": part[TARGET].mean() * 100 if len(part) else 0.0,
        }
    return pd.DataFrame(rows).T


def save_splits(splits: dict[str, pd.DataFrame], folder: str | Path = SPLITS_DIR) -> list[Path]:
    """Sauvegarde chaque jeu au format Parquet (``<folder>/<nom>.parquet``)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, part in splits.items():
        path = folder / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths.append(path)
    return paths


def load_split(name: str, folder: str | Path = SPLITS_DIR) -> pd.DataFrame:
    """Charge un jeu sauvegardé (train, val, test ou drift)."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Jeu inconnu : {name!r}. Choix possibles : {SPLIT_NAMES}")
    path = Path(folder) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Exécutez d'abord notebooks/03_protocole_experimental.ipynb."
        )
    return pd.read_parquet(path)
