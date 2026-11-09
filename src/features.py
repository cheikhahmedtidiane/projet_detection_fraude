"""Feature engineering : construction des variables et pipelines de prétraitement.

Toutes les transformations sont encapsulées dans des objets scikit-learn afin
d'être appliquées **à l'identique** à l'entraînement, à l'évaluation et dans la
démo Streamlit (aucune logique dupliquée, aucun risque de divergence).

Le constructeur de variables (:class:`FraudFeatureBuilder`) est **sans état** :
chaque variable se calcule à partir de la seule transaction, sans statistique
apprise. Il ne peut donc pas provoquer de fuite de données. Les seules étapes
apprises (standardisation) sont ajustées uniquement sur le jeu d'entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Colonnes brutes nécessaires au calcul des variables
RAW_INPUT_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "oldbalanceOrig",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Tolérance (unités monétaires) pour considérer deux montants comme égaux
BALANCE_TOL = 0.01

# ---------------------------------------------------------------------------
# Jeux de variables
# ---------------------------------------------------------------------------
# Variables décrivant uniquement la transaction (type, montant, moment) :
# elles ne dépendent pas des mécanismes de mise à jour des soldes du simulateur.
TRANSACTION_FEATURES: list[str] = [
    "is_transfer",
    "log_amount",
    "heure_sin",
    "heure_cos",
    "is_night",
]

# Variables issues des soldes : très discriminantes ici, mais en partie dues à
# des artefacts du simulateur PaySim (voir notebook 02, sections 4 à 6).
BALANCE_FEATURES: list[str] = [
    "log_oldbalanceOrig",
    "log_newbalanceOrig",
    "log_oldbalanceDest",
    "log_newbalanceDest",
    "erreur_orig",
    "erreur_dest",
    "compte_vide",
    "orig_solde_nul",
    "dest_soldes_nuls",
    "ratio_montant_solde",
]

FEATURE_SETS: dict[str, list[str]] = {
    "complet": TRANSACTION_FEATURES + BALANCE_FEATURES,
    "transaction": TRANSACTION_FEATURES,
}


def signed_log1p(x: pd.Series) -> pd.Series:
    """log(1 + |x|) en conservant le signe : compresse les valeurs extrêmes
    tout en gardant l'information de sens (utile pour les erreurs de solde)."""
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------
class FraudFeatureBuilder(BaseEstimator, TransformerMixin):
    """Construit les variables explicatives à partir des colonnes brutes.

    Transformer **sans état** : ``fit`` ne fait que vérifier les colonnes.

    Variables produites
    -------------------
    Transaction :
        * ``is_transfer`` : 1 si TRANSFER, 0 si CASH_OUT ;
        * ``log_amount`` : log(1 + montant) ;
        * ``heure_sin`` / ``heure_cos`` : encodage cyclique de l'heure
          (23h et 0h sont proches, ce qu'un entier 0-23 ne traduit pas) ;
        * ``is_night`` : 1 entre 0h et 6h inclus (H5).
    Soldes :
        * ``log_*balance*`` : soldes en échelle log ;
        * ``erreur_orig`` / ``erreur_dest`` : écart entre solde attendu et
          observé, en log signé (H4) ;
        * ``compte_vide`` : montant égal au solde initial non nul (H3) ;
        * ``orig_solde_nul`` : solde émetteur initial nul ;
        * ``dest_soldes_nuls`` : soldes destinataire nuls avant et après ;
        * ``ratio_montant_solde`` : montant / (solde initial + 1), en log.
    """

    def fit(self, X: pd.DataFrame, y=None) -> FraudFeatureBuilder:
        self._check_columns(X)
        self.feature_names_out_ = np.array(FEATURE_SETS["complet"], dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        amount = X["amount"].astype("float64")
        old_o = X["oldbalanceOrig"].astype("float64")
        new_o = X["newbalanceOrig"].astype("float64")
        old_d = X["oldbalanceDest"].astype("float64")
        new_d = X["newbalanceDest"].astype("float64")
        heure = X["step"].astype("int64") % 24

        out = pd.DataFrame(index=X.index)
        # --- transaction
        out["is_transfer"] = (X["type"].astype(str) == "TRANSFER").astype("int8")
        out["log_amount"] = np.log1p(amount)
        out["heure_sin"] = np.sin(2 * np.pi * heure / 24)
        out["heure_cos"] = np.cos(2 * np.pi * heure / 24)
        out["is_night"] = heure.between(0, 6).astype("int8")
        # --- soldes
        out["log_oldbalanceOrig"] = np.log1p(old_o)
        out["log_newbalanceOrig"] = np.log1p(new_o)
        out["log_oldbalanceDest"] = np.log1p(old_d)
        out["log_newbalanceDest"] = np.log1p(new_d)
        out["erreur_orig"] = signed_log1p(new_o + amount - old_o)
        out["erreur_dest"] = signed_log1p(old_d + amount - new_d)
        out["compte_vide"] = ((old_o > 0) & ((amount - old_o).abs() <= BALANCE_TOL)).astype("int8")
        out["orig_solde_nul"] = (old_o == 0).astype("int8")
        out["dest_soldes_nuls"] = ((old_d == 0) & (new_d == 0) & (amount > 0)).astype("int8")
        out["ratio_montant_solde"] = np.log1p(amount / (old_o + 1))
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_SETS["complet"], dtype=object)

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = set(RAW_INPUT_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes pour le feature engineering : {sorted(missing)}")


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Sélectionne un sous-ensemble de colonnes (dans l'ordre donné)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> ColumnSelector:
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Colonnes absentes : {sorted(missing)}")
        # Attribut « appris » (suffixe _) : permet à scikit-learn de savoir
        # que le transformer a été ajusté (check_is_fitted).
        self.columns_ = list(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns, dtype=object)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(feature_set: str = "complet", scale: bool = False) -> Pipeline:
    """Construit le pipeline de prétraitement.

    Args:
        feature_set: ``"complet"`` (toutes les variables) ou ``"transaction"``
            (sans variables de soldes, pour l'étude de robustesse).
        scale: ajoute une standardisation (utile pour les modèles linéaires ;
            inutile pour les modèles à base d'arbres).

    Returns:
        Pipeline scikit-learn produisant un DataFrame pandas.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Jeu de variables inconnu : {feature_set!r}. Choix : {list(FEATURE_SETS)}")

    steps: list[tuple[str, object]] = [
        ("features", FraudFeatureBuilder()),
        ("selection", ColumnSelector(FEATURE_SETS[feature_set])),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")
