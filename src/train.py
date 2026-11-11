"""Pipeline d'entraînement de bout en bout.

Reproduit en une seule commande, sans notebook, l'entraînement des modèles
retenus aux notebooks 06 et 07 :

    données brutes → contrôle du schéma → types TRANSFER / CASH_OUT
    → découpage temporel → entraînement (train) → seuil optimal (validation)
    → sauvegarde du modèle + métadonnées JSON

Usage ::

    python -m src.train                          # les deux modèles
    python -m src.train --models lightgbm_transaction
    python -m src.train --cost-fp 2000           # autre coût de fausse alerte

Le **test** et le **drift** ne sont pas utilisés ici : ils servent uniquement
à l'évaluation finale (notebook 07), jamais à l'entraînement ni au réglage.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn

from src.config import COST_FALSE_POSITIVE, MODELS_DIR, RANDOM_STATE, TARGET
from src.data import load_data, validate_schema
from src.evaluate import compute_metrics, optimal_threshold
from src.features import FEATURE_SETS
from src.models import make_model, positive_weight
from src.split import check_no_temporal_overlap, filter_fraud_types, temporal_split

logger = logging.getLogger("train")


@dataclass(frozen=True)
class ModelSpec:
    """Configuration complète d'un modèle à entraîner."""

    name: str
    model: str
    feature_set: str
    imbalance: str
    params: dict = field(default_factory=dict)
    description: str = ""


# Modèles retenus : réglages issus des notebooks 06 (choix, réglage) et 07 (seuil).
MODEL_SPECS: dict[str, ModelSpec] = {
    "lightgbm_complet": ModelSpec(
        name="lightgbm_complet",
        model="lightgbm",
        feature_set="complet",
        imbalance="ponderation",
        params={},
        description="LightGBM pondéré, toutes les variables (dont soldes).",
    ),
    "lightgbm_transaction": ModelSpec(
        name="lightgbm_transaction",
        model="lightgbm",
        feature_set="transaction",
        imbalance="ponderation",
        # Meilleure configuration de la validation croisée temporelle (notebook 06)
        params={
            "colsample_bytree": 1.0,
            "learning_rate": 0.05477661047824165,
            "min_child_weight": 0.1241480428274395,
            "n_estimators": 200,
            "num_leaves": 15,
            "reg_lambda": 0.06295301484516132,
            "subsample": 0.6975751180009072,
        },
        description="LightGBM pondéré et réglé, variables de transaction uniquement.",
    ),
}

# Métriques de validation conservées dans les métadonnées
_KEPT_METRICS = ["pr_auc", "roc_auc", "precision", "rappel", "f2", "n_alertes", "vp", "fp", "fn",
                 "montant_detecte_pct", "cout_total"]


def prepare_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Contrôle le schéma, garde TRANSFER / CASH_OUT et découpe dans le temps."""
    validate_schema(df)
    splits = temporal_split(filter_fraud_types(df))
    check_no_temporal_overlap(splits)
    for name, part in splits.items():
        logger.info("%-5s : %9s transactions, %5s fraudes", name, f"{len(part):,}",
                    f"{int(part[TARGET].sum()):,}")
    return splits


def train_one(
    spec: ModelSpec,
    train: pd.DataFrame,
    val: pd.DataFrame,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> tuple[object, dict]:
    """Entraîne un modèle sur le train et choisit son seuil sur la validation.

    Returns:
        ``(pipeline, metadata)`` : pipeline complet (données brutes → score) et
        métadonnées (réglages, seuil, métriques de validation, environnement).
    """
    y_train, y_val = train[TARGET].to_numpy(), val[TARGET].to_numpy()
    pos_weight = positive_weight(y_train)
    pipeline = make_model(spec.model, spec.feature_set, spec.imbalance,
                          pos_weight=pos_weight, **spec.params)

    start = time.perf_counter()
    pipeline.fit(train, y_train)
    duration = time.perf_counter() - start

    scores = pipeline.predict_proba(val)[:, 1]
    amounts = val["amount"].to_numpy()
    threshold = optimal_threshold(y_val, scores, amounts, cost_fp)
    metrics = compute_metrics(y_val, scores, threshold, amounts, cost_fp)
    logger.info("%s : entraîné en %.1f s | PR-AUC val = %.4f | seuil = %.4f | rappel = %.1f %% "
                "| montant détecté = %.1f %%", spec.name, duration, metrics["pr_auc"], threshold,
                100 * metrics["rappel"], metrics["montant_detecte_pct"])

    metadata = {
        "modele": asdict(spec),
        "variables": FEATURE_SETS[spec.feature_set],
        "poids_fraudes": pos_weight,
        "seuil": threshold,
        "cout_fausse_alerte": cost_fp,
        "validation": {k: float(metrics[k]) for k in _KEPT_METRICS},
        "entrainement": {
            "n_transactions": int(len(train)),
            "n_fraudes": int(y_train.sum()),
            "jours": [int(train["step"].min() // 24), int(train["step"].max() // 24)],
            "duree_s": round(duration, 2),
        },
        "environnement": {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "random_state": RANDOM_STATE,
        },
    }
    return pipeline, metadata


def save_model(pipeline, metadata: dict, output_dir: str | Path = MODELS_DIR) -> tuple[Path, Path]:
    """Sauvegarde le pipeline (``.joblib``) et ses métadonnées (``.json``)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = metadata["modele"]["name"]
    model_path = output_dir / f"{name}.joblib"
    meta_path = output_dir / f"{name}.json"
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_path, meta_path


def load_model(name: str, folder: str | Path = MODELS_DIR) -> tuple[object, dict]:
    """Recharge un modèle entraîné et ses métadonnées (utilisé par la démo)."""
    folder = Path(folder)
    pipeline = joblib.load(folder / f"{name}.joblib")
    metadata = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    return pipeline, metadata


def run_training(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    cost_fp: float = COST_FALSE_POSITIVE,
    output_dir: str | Path | None = MODELS_DIR,
) -> dict[str, dict]:
    """Exécute le pipeline complet sur un DataFrame brut.

    Args:
        df: données brutes (schéma du jeu Kaggle).
        model_names: modèles à entraîner (par défaut : tous ceux de :data:`MODEL_SPECS`).
        cost_fp: coût d'une fausse alerte pour le choix du seuil.
        output_dir: dossier de sauvegarde ; ``None`` pour ne rien écrire.

    Returns:
        {nom du modèle: métadonnées}.
    """
    model_names = model_names or list(MODEL_SPECS)
    unknown = set(model_names) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"Modèles inconnus : {sorted(unknown)}. Choix : {list(MODEL_SPECS)}")

    splits = prepare_splits(df)
    results = {}
    for name in model_names:
        pipeline, metadata = train_one(MODEL_SPECS[name], splits["train"], splits["val"], cost_fp)
        if output_dir is not None:
            model_path, _ = save_model(pipeline, metadata, output_dir)
            logger.info("%s : sauvegardé dans %s/", name, Path(model_path).parent.name)
        results[name] = metadata
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Entraîne les modèles de détection de fraude retenus.",
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS),
                        help="modèles à entraîner (défaut : tous)")
    parser.add_argument("--cost-fp", type=float, default=COST_FALSE_POSITIVE,
                        help=f"coût d'une fausse alerte pour le seuil (défaut : {COST_FALSE_POSITIVE})")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR,
                        help="dossier de sauvegarde des modèles (défaut : models/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="relire le CSV brut au lieu du cache Parquet")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Accents lisibles quelle que soit la console (Windows : cp1252 par défaut)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    start = time.perf_counter()
    logger.info("Chargement des données")
    df = load_data(use_cache=not args.no_cache)
    run_training(df, args.models, args.cost_fp, args.output_dir)
    logger.info("Terminé en %.1f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
