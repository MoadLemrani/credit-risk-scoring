"""Project-wide constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "polish_bankruptcy.csv.gz"
ARTIFACTS = Path(os.environ.get("CREDITRISK_ARTIFACTS", ROOT / "artifacts"))

FEATURES = [f"X{i}" for i in range(1, 65)]
TARGET = "bankrupt"
SEED = 42
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "credit-risk-scoring"
