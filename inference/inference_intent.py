# file: infer_intent.py

import os
import joblib

MODEL_DIR = "./models"

# Load encoder, classifier, and label map
encoder = joblib.load(os.path.join(MODEL_DIR, "minilm_encoder.joblib"))
label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.joblib"))

# Choose which model to use for inference
MODEL_NAME = "LogisticRegression"   # or GaussianNB, LinearSVC, etc.
clf = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}.joblib"))

def predict_intent(texts):
    # Accept single string or list of strings
    if isinstance(texts, str):
        texts = [texts]

    # Embed using the SAME MiniLM model used during training
    embeddings = encoder.encode(texts, show_progress_bar=False)

    # Predict numeric labels
    numeric_preds = clf.predict(embeddings)

    # Convert numeric → human-readable labels
    human_preds = [label_map[p] for p in numeric_preds]

    # Return both for debugging or logging
    return list(zip(texts, numeric_preds, human_preds))


if __name__ == "__main__":
    samples = [
        "I want to check my claim status",
        "How do I renew my policy?",
        "Can I update my address?",
        "how would you say fly in italian"
    ]

    results = predict_intent(samples)

    for text, num_label, human_label in results:
        print(f"Text: {text}")
        print(f"Predicted numeric label: {num_label}")
        print(f"Predicted intent: {human_label}")
        print()
