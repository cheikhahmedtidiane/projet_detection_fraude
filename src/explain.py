"""Explicabilité des modèles par les valeurs de Shapley (SHAP).

SHAP (Lundberg & Lee, 2017) décompose chaque prédiction en une somme de
contributions, une par variable :

    score(x) = valeur de base + somme des contributions SHAP de x

Pour les modèles à base d'arbres, ``TreeExplainer`` calcule ces valeurs
**exactement** et rapidement. Les contributions sont exprimées dans l'échelle
du score brut du modèle (log-odds) : une contribution positive pousse vers
« fraude », une contribution négative vers « légitime ».

Contrairement aux coefficients d'une régression logistique (notebook 05),
les valeurs SHAP restent interprétables quand les variables sont corrélées et
captent les effets non linéaires et les interactions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap

from src.viz import COLOR_FRAUD, COLOR_LEGIT, TEXT_PRIMARY, TEXT_SECONDARY

# Étapes qui ne transforment pas les données au moment de la prédiction
_NON_TRANSFORM_STEPS = {"sampler", "model"}


def transform_features(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Applique les étapes de prétraitement d'un pipeline entraîné.

    Le rééchantillonnage (``sampler``) est ignoré : il n'intervient qu'à
    l'entraînement. Le résultat est exactement ce que voit le modèle.
    """
    for name, step in pipeline.steps:
        if name not in _NON_TRANSFORM_STEPS:
            X = step.transform(X)
    return X


def compute_shap(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Valeurs SHAP exactes du modèle final d'un pipeline à base d'arbres.

    Args:
        pipeline: pipeline entraîné (étape finale nommée ``"model"``).
        X: données **brutes** (avant feature engineering).

    Returns:
        ``(shap_values, features, base_value)`` : contributions SHAP (une
        colonne par variable, même index que ``X``), variables vues par le
        modèle, et valeur de base (score moyen de référence, en log-odds).
    """
    features = transform_features(pipeline, X)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = explainer.shap_values(features)
    values = np.asarray(values)
    if values.ndim == 3:  # format (classes, n, p) ou (n, p, classes) selon la version
        values = values[1] if values.shape[0] == 2 else values[..., 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return pd.DataFrame(values, index=features.index, columns=features.columns), features, base


def mean_abs_shap(shap_values: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    """Importance globale : moyenne (éventuellement pondérée) des |SHAP|.

    Les poids permettent de corriger un échantillon stratifié (toutes les
    fraudes + une partie des légitimes) pour retrouver l'importance sur la
    population réelle.
    """
    importance = np.average(np.abs(shap_values.to_numpy()), axis=0, weights=weights)
    return pd.Series(importance, index=shap_values.columns).sort_values(ascending=False)


def group_share(shap_values: pd.DataFrame, groups: dict[str, list[str]],
                weights: np.ndarray | None = None) -> pd.Series:
    """Part de l'importance totale (somme des |SHAP|) portée par chaque groupe de variables."""
    importance = mean_abs_shap(shap_values, weights)
    total = importance.sum()
    return pd.Series({g: importance[cols].sum() / total for g, cols in groups.items()})


def plot_contributions(ax, shap_row: pd.Series, feature_row: pd.Series, base_value: float,
                       max_features: int = 8, title: str | None = None) -> None:
    """Explication locale d'une prédiction : contributions SHAP triées.

    Les barres orange poussent vers « fraude », les bleues vers « légitime »
    (code couleur du projet). Les variables les moins influentes sont
    regroupées dans « autres variables ».
    """
    order = shap_row.abs().sort_values(ascending=False).index
    top, rest = order[:max_features], order[max_features:]
    contrib = shap_row[top]
    labels = [f"{f} = {_fmt(feature_row[f])}" for f in top]
    if len(rest):
        contrib = pd.concat([contrib, pd.Series({"autres": shap_row[rest].sum()})])
        labels.append(f"{len(rest)} autres variables")

    y_pos = np.arange(len(contrib))[::-1]
    colors = [COLOR_FRAUD if v > 0 else COLOR_LEGIT for v in contrib]
    ax.barh(y_pos, contrib.to_numpy(), color=colors, height=0.6)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(y_pos, labels)
    span = np.abs(contrib.to_numpy()).max()
    for y, v in zip(y_pos, contrib.to_numpy()):
        ax.text(v + np.sign(v) * span * 0.02, y, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlim(-span * 1.3, span * 1.3)
    score = base_value + shap_row.sum()
    proba = 1 / (1 + np.exp(-score))
    ax.set_xlabel(f"Contribution SHAP (log-odds) — base {base_value:.2f}, "
                  f"score {score:.2f}, probabilité {proba:.3f}")
    if title:
        ax.set_title(title, loc="left")


def _fmt(value) -> str:
    value = float(value)
    return f"{value:.0f}" if value == int(value) else f"{value:.2f}"
