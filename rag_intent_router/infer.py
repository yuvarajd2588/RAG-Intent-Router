# rag_intent_router/infer.py

import os
import joblib

PACKAGE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PACKAGE_DIR, "models")

encoder = joblib.load(os.path.join(MODEL_DIR, "minilm_encoder.joblib"))
label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "LogisticRegression.joblib"))

def infer_intent(text):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    embeddings = encoder.encode(texts, show_progress_bar=False)
    numeric_preds = clf.predict(embeddings)
    human_preds = [label_map[p] for p in numeric_preds]

    return human_preds
