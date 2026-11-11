"""Fonctions partagées par les pages de la démo.

Toute la logique métier (variables, modèles, seuil, SHAP) vient de ``src/`` :
la démo ne fait que l'appeler. Les chargements coûteux sont mis en cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SPLITS_DIR, TARGET  # noqa: E402
from src.evaluate import cost_curve  # noqa: E402
from src.explain import compute_shap  # noqa: E402
from src.split import load_split  # noqa: E402
from src.train import MODEL_SPECS, load_model  # noqa: E402

MODEL_NAMES = list(MODEL_SPECS)
MODEL_LABELS = {
    "lightgbm_transaction": "Sans les soldes (réaliste)",
    "lightgbm_complet": "Avec les soldes (artefacts)",
}
SPLIT_LABELS = {"val": "Validation (jours 13-14)", "test": "Test (jours 15-16)"}
COST_FP_CHOICES = [100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]


def fmt_money(x: float) -> str:
    """Montant lisible : 1 234 567 → « 1,23 M »."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.2f} M".replace(",", " ").replace(".", ",")
    return f"{x:,.0f}".replace(",", " ")


def fmt_int(x: float) -> str:
    return f"{int(x):,}".replace(",", " ")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f} %".replace(".", ",")


# ---------------------------------------------------------------------------
# Chargements (mis en cache)
# ---------------------------------------------------------------------------
def models_available() -> bool:
    return all((MODELS_DIR / f"{n}.joblib").exists() and (MODELS_DIR / f"{n}.json").exists()
               for n in MODEL_NAMES)


def splits_available() -> bool:
    return all((SPLITS_DIR / f"{s}.parquet").exists() for s in SPLIT_LABELS)


def require_models() -> None:
    """Arrête la page avec un message clair si les modèles n'ont pas été entraînés."""
    if not models_available():
        st.error("Les modèles entraînés sont introuvables dans `models/`.")
        st.markdown("Entraînez-les d'abord (≈ 30 secondes) depuis la racine du projet :")
        st.code("python -m src.train", language="bash")
        st.stop()


def require_splits() -> None:
    if not splits_available():
        st.error("Les jeux de données découpés sont introuvables dans `data/processed/splits/`.")
        st.markdown("Exécutez le notebook `03_protocole_experimental.ipynb` pour les générer.")
        st.stop()


@st.cache_resource(show_spinner="Chargement des modèles…")
def get_models() -> dict[str, tuple[object, dict]]:
    """{nom: (pipeline, métadonnées)} pour chaque modèle retenu."""
    return {name: load_model(name) for name in MODEL_NAMES}


@st.cache_data(show_spinner="Calcul des scores sur le jeu de données…")
def get_scored_split(split: str) -> pd.DataFrame:
    """Jeu de validation ou de test, avec le score de chaque modèle."""
    df = load_split(split)
    cols = ["step", "type", "amount", "oldbalanceOrig", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", TARGET]
    out = df[cols].copy()
    out["type"] = out["type"].astype(str)
    out["jour"] = out["step"] // 24
    out["heure"] = out["step"] % 24
    for name, (pipeline, _) in get_models().items():
        out[f"score_{name}"] = pipeline.predict_proba(df)[:, 1]
    return out


@st.cache_data(show_spinner=False)
def optimal_threshold_for(model: str, cost_fp: float) -> dict[str, float]:
    """Seuil qui minimise le coût sur la **validation** pour un coût de fausse alerte donné."""
    val = get_scored_split("val")
    curve = cost_curve(val[TARGET].to_numpy(), val[f"score_{model}"].to_numpy(),
                       val["amount"].to_numpy(), cost_fp)
    best = curve.loc[curve["cout_total"].idxmin()]
    return {"seuil": float(best["seuil"]), "rappel": float(best["rappel"]),
            "precision": float(best["precision"]), "n_alertes": int(best["n_alertes"])}


@st.cache_data(show_spinner=False)
def load_results(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{name}.csv"
    return pd.read_csv(path, index_col="modele") if path.exists() else None


def figure_path(name: str) -> Path | None:
    path = FIGURES_DIR / f"{name}.png"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Transaction unique
# ---------------------------------------------------------------------------
def make_transaction(type_: str, amount: float, hour: int, old_orig: float, new_orig: float,
                     old_dest: float, new_dest: float) -> pd.DataFrame:
    """Construit une transaction brute au format attendu par les pipelines."""
    return pd.DataFrame([{
        "step": int(hour),
        "type": type_,
        "amount": float(amount),
        "oldbalanceOrig": float(old_orig),
        "newbalanceOrig": float(new_orig),
        "oldbalanceDest": float(old_dest),
        "newbalanceDest": float(new_dest),
    }])


def score_and_explain(model: str, tx: pd.DataFrame) -> dict:
    """Score, décision et contributions SHAP d'une transaction pour un modèle."""
    pipeline, meta = get_models()[model]
    proba = float(pipeline.predict_proba(tx)[0, 1])
    values, feats, base = compute_shap(pipeline, tx)
    return {"proba": proba, "seuil": float(meta["seuil"]), "alerte": proba >= meta["seuil"],
            "shap": values.iloc[0], "features": feats.iloc[0], "base": base}


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


__all__ = ["np", "pd", "st", "ROOT"]
