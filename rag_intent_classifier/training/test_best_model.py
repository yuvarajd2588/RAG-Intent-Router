# file: test_best_model.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"

TEST_PATH = DATA_DIR / "hugging_face/plus/clinc_test_with_labels.csv"

MODEL_NAME = "LogisticRegression"

test_df = pd.read_csv(TEST_PATH)

text_col = "text"
numeric_label_col = "intent"

X_test_text, y_test = test_df[text_col], test_df[numeric_label_col]

encoder = joblib.load(MODEL_DIR / "minilm_encoder.joblib")
clf = joblib.load(MODEL_DIR / f"{MODEL_NAME}.joblib")
label_map = joblib.load(MODEL_DIR / "label_map.joblib")

print("Encoding test...")
X_test = encoder.encode(X_test_text.tolist(), show_progress_bar=True)

preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
weighted_f1 = f1_score(y_test, preds, average="weighted")

print("\n=== Test Results ===")
print(f"Model: {MODEL_NAME}")
print(f"Accuracy: {acc}")
print(f"Macro F1: {macro_f1}")
print(f"Weighted F1: {weighted_f1}")

human_preds = [label_map[p] for p in preds]

pd.DataFrame(
    list(zip(preds, human_preds)),
    columns=["numeric_label", "human_label"],
).to_csv(MODEL_DIR / "test_predictions.csv", index=False)
