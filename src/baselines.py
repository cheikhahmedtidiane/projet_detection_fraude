"""Modèles de référence (baselines).

Un modèle de Machine Learning n'a d'intérêt que s'il fait mieux que des
approches simples. Trois références sont définies :

* ``RegleIsFlagged`` : la règle métier existante du jeu de données
  (``isFlaggedFraud``) ;
* ``RegleCompteVide`` : la règle « montant = solde initial > 0 » issue de
  l'hypothèse H3 ;
* ``make_logistic_regression`` : une régression logistique pondérée, premier
  modèle appris.

Les règles respectent l'interface scikit-learn (``fit`` / ``predict_proba`` /
``predict``) pour être évaluées exactement comme les modèles appris.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features import BALANCE_TOL, build_preprocessor


class _RuleClassifier(BaseEstimator, ClassifierMixin):
    """Classe de base d'une règle déterministe : rien à apprendre."""

    def fit(self, X: pd.DataFrame, y=None) -> _RuleClassifier:
        self.classes_ = np.array([0, 1])
        return self

    def _rule(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        flag = self._rule(X).astype("float64")
        return np.column_stack([1 - flag, flag])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._rule(X).astype(int)


class RegleIsFlagged(_RuleClassifier):
    """Règle métier existante : alerte si ``isFlaggedFraud`` vaut 1."""

    def _rule(self, X: pd.DataFrame) -> np.ndarray:
        return X["isFlaggedFraud"].to_numpy() == 1


class RegleCompteVide(_RuleClassifier):
    """Alerte si la transaction vide le compte émetteur (montant = solde > 0)."""

    def _rule(self, X: pd.DataFrame) -> np.ndarray:
        amount = X["amount"].to_numpy(dtype="float64")
        old = X["oldbalanceOrig"].to_numpy(dtype="float64")
        return (old > 0) & (np.abs(amount - old) <= BALANCE_TOL)


def make_logistic_regression(feature_set: str = "complet", C: float = 1.0) -> Pipeline:
    """Régression logistique avec prétraitement et pondération des classes.

    * standardisation des variables (indispensable pour un modèle linéaire
      régularisé) ;
    * ``class_weight="balanced"`` : chaque fraude pèse autant que ~550
      transactions légitimes dans la fonction de perte, ce qui compense le
      déséquilibre sans rééchantillonner ;
    * régularisation L2 (``C`` = inverse de la force de régularisation), qui
      limite l'effet de la colinéarité entre variables.
    """
    return Pipeline(
        [
            ("preprocessing", build_preprocessor(feature_set, scale=True)),
            (
                "model",
                LogisticRegression(
                    C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
