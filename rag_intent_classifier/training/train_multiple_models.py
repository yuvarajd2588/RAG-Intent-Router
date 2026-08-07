# file: train_with_validation.py

from pathlib import Path

import pandas as pd
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sentence_transformers import SentenceTransformer
import joblib

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "hugging_face/plus/clinc_train_with_labels.csv"
VAL_PATH = DATA_DIR / "hugging_face/plus/clinc_val_with_labels.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

text_col = "text"
numeric_label_col = "intent"
human_label_col = "intent_label"

X_train_text, y_train = train_df[text_col], train_df[numeric_label_col]

# Build label map automatically
label_map = dict(zip(train_df[numeric_label_col], train_df[human_label_col]))
joblib.dump(label_map, MODEL_DIR / "label_map.joblib")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encoding train...")
X_train = encoder.encode(X_train_text.tolist(), show_progress_bar=True)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1, C=3.0),
    "GaussianNB": GaussianNB(),
    "LinearSVC": LinearSVC(C=1.0, max_iter=2000),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000),
    "SGDClassifier": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3),
    "RidgeClassifier": RidgeClassifier(),
}

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_DIR / f"{name}.joblib")

print("\nTraining complete. Model files written to", MODEL_DIR)
