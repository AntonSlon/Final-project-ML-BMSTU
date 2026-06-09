from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_P2P_PATH = RAW_DATA_DIR / "final_p2p_log.csv.gz"
RAW_TRANS_PATH = RAW_DATA_DIR / "final_trans_log.csv.gz"

MODEL_PATH = MODELS_DIR
METRICS_PATH = REPORTS_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.2
