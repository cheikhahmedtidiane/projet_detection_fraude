"""Configuration centrale du projet : chemins, constantes et paramètres globaux."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV_PATH = RAW_DIR / "Synthetic_Financial_datasets_log.csv"
CLEAN_PARQUET_PATH = PROCESSED_DIR / "transactions.parquet"

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Colonnes
# ---------------------------------------------------------------------------
TARGET = "isFraud"
RULE_FLAG = "isFlaggedFraud"
TIME_COL = "step"

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Seuls types contenant de la fraude (validé en EDA, hypothèse H1)
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# ---------------------------------------------------------------------------
# Protocole expérimental : découpage temporel (en jours, bornes incluses)
# ---------------------------------------------------------------------------
# Jour = step // 24. Voir notebooks/03_protocole_experimental.ipynb.
SPLITS_DIR = PROCESSED_DIR / "splits"
TRAIN_LAST_DAY = 12  # train : jours 0 à 12
VAL_LAST_DAY = 14  # validation : jours 13 à 14
TEST_LAST_DAY = 16  # test : jours 15 à 16
# drift : jours 17 et suivants (régime d'activité anormal du simulateur)
SPLIT_NAMES = ["train", "val", "test", "drift"]

# ---------------------------------------------------------------------------
# Évaluation : coût métier
# ---------------------------------------------------------------------------
# Coût d'une fausse alerte (investigation manuelle + friction client), en unités
# monétaires du jeu de données. Hypothèse : ~3 % du montant médian d'une
# transaction TRANSFER/CASH_OUT (~175 000). Une analyse de sensibilité sera
# menée lors du choix du seuil.
COST_FALSE_POSITIVE = 5_000

RESULTS_DIR = REPORTS_DIR / "results"
