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
