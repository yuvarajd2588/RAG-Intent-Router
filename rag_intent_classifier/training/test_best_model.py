# file: test_best_model.py

from pathlib import Path

import joblib

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"

MODEL_NAME = "LogisticRegression"

encoder = joblib.load(MODEL_DIR / "minilm_encoder.joblib")
clf = joblib.load(MODEL_DIR / f"{MODEL_NAME}.joblib")

print(f"Loaded encoder and {MODEL_NAME} classifier from {MODEL_DIR}")
