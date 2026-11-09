"""Évaluation des modèles : métriques statistiques et métriques métier.

Toutes les évaluations du projet passent par :func:`compute_metrics`, afin que
les modèles soient comparés exactement de la même manière.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import COST_FALSE_POSITIVE, RESULTS_DIR

# Couleurs des séries « modèles » : palette catégorielle dans un ordre fixe.
# L'orange est volontairement exclu : il signifie « fraude » dans tout le projet.
MODEL_COLORS = ["#2a78d6", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#008300"]


def business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> dict[str, float]:
    """Coût métier d'un ensemble de décisions.

    * fraude ratée (faux négatif) : le montant est perdu ;
    * fausse alerte (faux positif) : coût fixe ``cost_fp``.

    Returns:
        Coût des fraudes ratées, coût des fausses alertes, coût total et part
        du montant frauduleux détecté.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    amounts = np.asarray(amounts, dtype="float64")

    fraud_amount = amounts[y_true].sum()
    missed = amounts[y_true & ~y_pred].sum()
    n_fp = int((~y_true & y_pred).sum())
    return {
        "cout_fraudes_ratees": float(missed),
        "cout_fausses_alertes": float(n_fp * cost_fp),
        "cout_total": float(missed + n_fp * cost_fp),
        "montant_detecte_pct": float((1 - missed / fraud_amount) * 100) if fraud_amount else 0.0,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    amounts: np.ndarray | None = None,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> dict[str, float]:
    """Calcule l'ensemble des métriques d'évaluation d'un modèle.

    Args:
        y_true: vraie classe (0/1).
        y_score: score ou probabilité de fraude (plus élevé = plus suspect).
        threshold: seuil de décision (fraude si score >= seuil).
        amounts: montants des transactions (requis pour le coût métier).
        cost_fp: coût d'une fausse alerte.

    Returns:
        Dictionnaire de métriques :
        * indépendantes du seuil : ``pr_auc`` (métrique principale), ``roc_auc`` ;
        * dépendantes du seuil : précision, rappel, F1, F2, matrice de confusion ;
        * métier (si ``amounts``) : coûts et part du montant frauduleux détecté.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype="float64")
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "seuil": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "rappel": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "n_alertes": int(tp + fp),
        "vp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "vn": int(tn),
    }
    if amounts is not None:
        metrics.update(business_cost(y_true, y_pred, amounts, cost_fp))
    return metrics


def compare_models(
    scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    thresholds: dict[str, float] | None = None,
    amounts: np.ndarray | None = None,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> pd.DataFrame:
    """Tableau comparatif de plusieurs modèles évalués sur les mêmes données.

    Args:
        scores: {nom du modèle: scores}.
        y_true: vraie classe.
        thresholds: seuil par modèle (0,5 par défaut).
        amounts: montants pour le coût métier.
    """
    thresholds = thresholds or {}
    rows = {
        name: compute_metrics(y_true, s, thresholds.get(name, 0.5), amounts, cost_fp)
        for name, s in scores.items()
    }
    return pd.DataFrame(rows).T


def bootstrap_pr_auc(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    reference: str | None = None,
    n_boot: int = 300,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Intervalles de confiance bootstrap de la PR-AUC, appariés entre modèles.

    Les mêmes rééchantillonnages (tirage avec remise des transactions) sont
    utilisés pour tous les modèles : les écarts entre modèles sont donc
    **appariés**, ce qui donne des comparaisons bien plus précises que deux
    intervalles calculés séparément.

    Limite : le bootstrap simple suppose des observations indépendantes ; il
    ignore la dépendance temporelle entre transactions proches.

    Args:
        y_true: vraie classe.
        scores: {nom du modèle: scores}.
        reference: modèle de référence ; si fourni, on ajoute l'intervalle de
            l'écart ``PR-AUC(modèle) - PR-AUC(référence)`` et la part des
            rééchantillonnages où le modèle fait mieux.
        n_boot: nombre de rééchantillonnages.
        alpha: 1 - niveau de confiance.

    Returns:
        DataFrame indexé par modèle : PR-AUC observée, bornes de l'intervalle
        et, le cas échéant, statistiques de l'écart à la référence.
    """
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot = {name: np.empty(n_boot) for name in scores}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        for name, s in scores.items():
            boot[name][b] = average_precision_score(y_b, np.asarray(s)[idx])

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = {}
    for name, s in scores.items():
        row = {
            "pr_auc": float(average_precision_score(y_true, s)),
            "ic_bas": float(np.percentile(boot[name], lo)),
            "ic_haut": float(np.percentile(boot[name], hi)),
        }
        if reference is not None:
            diff = boot[name] - boot[reference]
            row["ecart_ref"] = row["pr_auc"] - float(average_precision_score(y_true, scores[reference]))
            row["ecart_ic_bas"] = float(np.percentile(diff, lo))
            row["ecart_ic_haut"] = float(np.percentile(diff, hi))
            row["p_meilleur"] = float((diff > 0).mean())
        rows[name] = row
    return pd.DataFrame(rows).T


def plot_pr_curves(
    ax, scores: dict[str, np.ndarray], y_true: np.ndarray, legend_loc: str = "lower left"
) -> None:
    """Trace les courbes précision-rappel de plusieurs modèles sur un même axe."""
    y_true = np.asarray(y_true)
    for color, (name, s) in zip(MODEL_COLORS, scores.items()):
        precision, recall, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)
        ax.step(recall, precision, where="post", color=color, linewidth=2,
                label=f"{name} (PR-AUC = {ap:.3f})")
    prevalence = y_true.mean()
    ax.axhline(prevalence, color="#52514e", linestyle=":", linewidth=1,
               label=f"Hasard ({prevalence:.2%})")
    ax.set_xlabel("Rappel (part des fraudes détectées)")
    ax.set_ylabel("Précision (part d'alertes justifiées)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.legend(loc=legend_loc)


def save_results(df: pd.DataFrame, name: str, folder: str | Path = RESULTS_DIR) -> Path:
    """Sauvegarde un tableau de résultats en CSV (versionné, pour la traçabilité)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    df.to_csv(path, index_label="modele")
    return path
