# file: train_with_validation.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import joblib

DATA_DIR = "./data"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "hugging_face/plus/clinc_train_with_labels.csv")
VAL_PATH   = os.path.join(DATA_DIR, "hugging_face/plus/clinc_val_with_labels.csv")

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

text_col = "text"
numeric_label_col = "intent"
human_label_col = "intent_label"

X_train_text, y_train = train_df[text_col], train_df[numeric_label_col]
X_val_text,   y_val   = val_df[text_col],   val_df[numeric_label_col]

# Build label map automatically
label_map = dict(zip(train_df[numeric_label_col], train_df[human_label_col]))
joblib.dump(label_map, os.path.join(MODEL_DIR, "label_map.joblib"))

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encoding train...")
X_train = encoder.encode(X_train_text.tolist(), show_progress_bar=True)
print("Encoding validation...")
X_val   = encoder.encode(X_val_text.tolist(),   show_progress_bar=True)

joblib.dump(encoder, os.path.join(MODEL_DIR, "minilm_encoder.joblib"))

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1, C=3.0),
    "GaussianNB": GaussianNB(),
    "LinearSVC": LinearSVC(C=1.0, max_iter=2000),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000),
    "SGDClassifier": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3),
    "RidgeClassifier": RidgeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = []

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")
    weighted_f1 = f1_score(y_val, preds, average="weighted")

    results.append([name, acc, macro_f1, weighted_f1])

    joblib.dump(clf, os.path.join(MODEL_DIR, f"{name}.joblib"))

df_results = pd.DataFrame(results, columns=["Model", "Val Accuracy", "Val Macro F1", "Val Weighted F1"])
df_results.to_csv(os.path.join(MODEL_DIR, "validation_results.csv"), index=False)

print("\n=== Validation Results ===")
print(df_results.sort_values(by="Val Accuracy", ascending=False))
