"""Tests statistiques et tailles d'effet utilisés pour valider les hypothèses.

Avec plusieurs millions d'observations, presque toute différence devient
« statistiquement significative » (p-value ≈ 0). La p-value seule ne dit donc
rien de l'importance pratique d'un effet : chaque test est systématiquement
accompagné d'une **taille d'effet** interprétable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """Résultat normalisé d'un test statistique."""

    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect_size: float
    effect_label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Interprétation des tailles d'effet (seuils de Cohen, 1988)
# ---------------------------------------------------------------------------
def interpret_cramers_v(v: float, dof: int = 1) -> str:
    """Interprète le V de Cramér selon Cohen (seuils ajustés au ddl min)."""
    # Seuils pour ddl* = min(r-1, c-1) ; on ne retient que ddl* ≤ 3
    thresholds = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29)}
    small, medium, large = thresholds[min(max(dof, 1), 3)]
    if v < small:
        return "négligeable"
    if v < medium:
        return "faible"
    if v < large:
        return "moyen"
    return "fort"


def interpret_rank_biserial(r: float) -> str:
    """Interprète la corrélation rang-bisériale (|r|) selon Cohen."""
    r = abs(r)
    if r < 0.1:
        return "négligeable"
    if r < 0.3:
        return "faible"
    if r < 0.5:
        return "moyen"
    return "fort"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def chi2_independence(x: pd.Series, y: pd.Series) -> TestResult:
    """Test du χ² d'indépendance entre deux variables catégorielles + V de Cramér.

    Args:
        x: première variable catégorielle.
        y: seconde variable catégorielle (ex. la cible).

    Returns:
        Résultat du test avec le V de Cramér comme taille d'effet.
    """
    table = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    dof_min = min(table.shape) - 1
    v = float(np.sqrt(chi2 / (n * dof_min))) if dof_min > 0 else 0.0
    return TestResult(
        test="Chi² d'indépendance",
        statistic=float(chi2),
        p_value=float(p),
        effect_name="V de Cramér",
        effect_size=v,
        effect_label=interpret_cramers_v(v, dof_min),
    )


def mann_whitney(group_a: pd.Series, group_b: pd.Series) -> TestResult:
    """Test de Mann-Whitney U (bilatéral) + corrélation rang-bisériale.

    Test non paramétrique : il ne suppose pas la normalité des distributions,
    ce qui est adapté aux montants financiers, très asymétriques.

    La corrélation rang-bisériale r = 2·U/(n_a·n_b) − 1 vaut :
      * +1 si toutes les valeurs de A sont supérieures à celles de B ;
      * 0 en l'absence de différence ;
      * −1 si toutes les valeurs de A sont inférieures.

    Args:
        group_a: échantillon A (ex. fraudes).
        group_b: échantillon B (ex. transactions légitimes).
    """
    a = np.asarray(group_a, dtype="float64")
    b = np.asarray(group_b, dtype="float64")
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = float(2 * u / (len(a) * len(b)) - 1)
    return TestResult(
        test="Mann-Whitney U",
        statistic=float(u),
        p_value=float(p),
        effect_name="r rang-bisérial",
        effect_size=r,
        effect_label=interpret_rank_biserial(r),
    )


def odds_ratio(exposure: pd.Series, outcome: pd.Series) -> float:
    """Odds ratio d'un tableau 2×2 (exposition binaire × issue binaire).

    Une correction de Haldane (+0,5 par case) évite la division par zéro.
    """
    table = pd.crosstab(exposure.astype(bool), outcome.astype(bool))
    table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)
    a = table.loc[True, True] + 0.5
    b = table.loc[True, False] + 0.5
    c = table.loc[False, True] + 0.5
    d = table.loc[False, False] + 0.5
    return float((a * d) / (b * c))


def results_table(results: dict[str, TestResult], alpha: float = 0.05) -> pd.DataFrame:
    """Met en forme plusieurs résultats avec correction de Bonferroni.

    Args:
        results: dictionnaire {nom de l'hypothèse: résultat}.
        alpha: risque de première espèce global.
    """
    alpha_corr = alpha / len(results)
    df = pd.DataFrame({k: v.to_dict() for k, v in results.items()}).T
    df["significatif"] = df["p_value"].astype(float) < alpha_corr
    df.attrs["alpha_bonferroni"] = alpha_corr
    return df
