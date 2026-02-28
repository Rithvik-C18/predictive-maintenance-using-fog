from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

DATA_FILE = DATA_DIR / "predictive_maintenance.csv"
MODEL_FILE = MODEL_DIR / "logreg_model.joblib"

TARGET_COLUMN = "failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASS_WEIGHT = "balanced"
