"""Modèles avancés et stratégies de gestion du déséquilibre.

Trois familles de modèles à base d'arbres sont comparées :

* ``random_forest`` : forêt aléatoire (bagging), robuste et peu sensible aux
  hyperparamètres ;
* ``xgboost`` : gradient boosting régularisé (Chen & Guestrin, 2016) ;
* ``lightgbm`` : gradient boosting par histogrammes, croissance des arbres
  feuille par feuille, très rapide sur de gros volumes (Ke et al., 2017).

Quatre stratégies de gestion du déséquilibre (question de recherche RQ) :

* ``aucune`` : le modèle voit les données telles quelles ;
* ``ponderation`` : les fraudes pèsent ``pos_weight`` fois plus dans la perte ;
* ``sous_echantillonnage`` : on retire aléatoirement des transactions légitimes ;
* ``smote`` : on crée des fraudes synthétiques par interpolation entre voisins
  (Chawla et al., 2002).

Le rééchantillonnage est placé **dans** un pipeline ``imblearn`` : il n'est
appliqué qu'à l'entraînement (``fit``), jamais lors de la prédiction. Les jeux
de validation et de test gardent donc leur distribution réelle.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.features import build_preprocessor
from src.split import step_to_day

MODEL_NAMES: list[str] = ["random_forest", "xgboost", "lightgbm"]
IMBALANCE_STRATEGIES: list[str] = ["aucune", "ponderation", "sous_echantillonnage", "smote"]

# Libellés lisibles pour les tableaux et graphiques
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

# Hyperparamètres par défaut : raisonnables, volontairement non optimisés.
DEFAULT_PARAMS: dict[str, dict] = {
    "random_forest": {
        "n_estimators": 100,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        # Défaut LightGBM = 1e-3 : sur des données presque séparables (jeu
        # complet), les feuilles pures ont une hessienne quasi nulle et des
        # valeurs démesurées, ce qui fait diverger l'apprentissage (notebook 06).
        # On s'aligne sur la valeur par défaut de XGBoost.
        "min_child_weight": 1.0,
    },
}


def positive_weight(y: np.ndarray) -> float:
    """Rapport légitimes / fraudes : poids qui équilibre les deux classes."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("Aucune fraude dans y : impossible de calculer le poids.")
    return float((y == 0).sum() / n_pos)


def make_classifier(name: str, pos_weight: float = 1.0, **params):
    """Instancie un classifieur à base d'arbres.

    Args:
        name: ``"random_forest"``, ``"xgboost"`` ou ``"lightgbm"``.
        pos_weight: poids d'une fraude relativement à une transaction légitime
            (1 = pas de pondération).
        **params: hyperparamètres qui remplacent les valeurs par défaut.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Modèle inconnu : {name!r}. Choix : {MODEL_NAMES}")
    p = {**DEFAULT_PARAMS[name], **params}

    if name == "random_forest":
        class_weight = None if pos_weight == 1 else {0: 1.0, 1: pos_weight}
        return RandomForestClassifier(
            class_weight=class_weight, n_jobs=-1, random_state=RANDOM_STATE, **p
        )
    if name == "xgboost":
        return XGBClassifier(
            scale_pos_weight=pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **p,
        )
    # Avec une pondération, l'initialisation par défaut de LightGBM
    # (boost_from_average) part du log-odds de la prévalence brute (≈ -6,3) sans
    # tenir compte du poids des fraudes : le modèle démarre très loin de
    # l'optimum pondéré et les premiers pas de Newton sont démesurés, ce qui fait
    # s'effondrer la PR-AUC (diagnostic au notebook 06). On part donc de 0.
    p.setdefault("boost_from_average", pos_weight == 1)
    return LGBMClassifier(
        scale_pos_weight=pos_weight, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1, **p
    )


def make_sampler(strategy: str, sampling_ratio: float = 0.1):
    """Crée l'étape de rééchantillonnage (``None`` si la stratégie n'en a pas).

    Args:
        strategy: une des valeurs de :data:`IMBALANCE_STRATEGIES`.
        sampling_ratio: rapport fraudes / légitimes visé après rééchantillonnage
            (0,1 = une fraude pour dix légitimes).
    """
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy!r}. Choix : {IMBALANCE_STRATEGIES}")
    if strategy == "sous_echantillonnage":
        return RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=RANDOM_STATE)
    if strategy == "smote":
        return SMOTE(sampling_strategy=sampling_ratio, k_neighbors=5, random_state=RANDOM_STATE)
    return None


def make_model(
    name: str,
    feature_set: str = "complet",
    imbalance: str = "ponderation",
    pos_weight: float | None = None,
    sampling_ratio: float = 0.1,
    preprocess: bool = True,
    **params,
) -> Pipeline:
    """Construit le pipeline complet : prétraitement → [rééchantillonnage] → modèle.

    Args:
        name: famille de modèle (:data:`MODEL_NAMES`).
        feature_set: ``"complet"`` ou ``"transaction"``.
        imbalance: stratégie de gestion du déséquilibre.
        pos_weight: poids des fraudes, **obligatoire** pour ``"ponderation"``
            (en général :func:`positive_weight` du jeu d'entraînement).
        sampling_ratio: rapport fraudes / légitimes pour le rééchantillonnage.
        preprocess: ``False`` pour travailler sur des variables déjà calculées
            (accélère les expériences ; le constructeur de variables est sans
            état, le résultat est identique).
        **params: hyperparamètres du modèle.
    """
    sampler = make_sampler(imbalance, sampling_ratio)
    if imbalance == "ponderation":
        if pos_weight is None:
            raise ValueError("pos_weight est obligatoire avec la stratégie 'ponderation'.")
        weight = pos_weight
    else:
        weight = 1.0

    steps: list[tuple[str, object]] = []
    if preprocess:
        # imblearn refuse un Pipeline imbriqué : on reprend ses étapes à plat
        # (elles conservent leur sortie pandas, configurée par set_output).
        steps.extend(build_preprocessor(feature_set).steps)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_classifier(name, weight, **params)))
    return Pipeline(steps)


def temporal_cv_folds(
    steps: np.ndarray, n_folds: int = 3, val_days: int = 2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validation croisée temporelle à fenêtre croissante, par blocs de jours.

    Le dernier bloc de ``val_days`` jours sert de validation au dernier pli,
    le bloc précédent au pli d'avant, etc. Chaque pli s'entraîne sur **tous les
    jours antérieurs** à son bloc de validation : le modèle ne voit jamais le
    futur, comme dans le protocole principal.

    Exemple pour 13 jours (0-12), 3 plis de 2 jours :
    ``[0-6] → [7-8]``, ``[0-8] → [9-10]``, ``[0-10] → [11-12]``.

    Returns:
        Liste de couples (indices d'entraînement, indices de validation),
        directement utilisable comme argument ``cv`` de scikit-learn.
    """
    days = step_to_day(np.asarray(steps))
    last_day = int(days.max())
    first_val_day = last_day - n_folds * val_days + 1
    if first_val_day <= int(days.min()):
        raise ValueError("Pas assez de jours pour ce nombre de plis.")

    folds = []
    for k in range(n_folds):
        start = first_val_day + k * val_days
        train_idx = np.flatnonzero(days < start)
        val_idx = np.flatnonzero((days >= start) & (days < start + val_days))
        folds.append((train_idx, val_idx))
    return folds
