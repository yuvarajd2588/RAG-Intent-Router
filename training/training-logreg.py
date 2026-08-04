import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------------------------------------------------
# 1. Load your prepared CSV files
# ---------------------------------------------------------
train_df = pd.read_csv("data/hugging_face/plus/clinc_train_with_labels.csv")
val_df = pd.read_csv("data/hugging_face/plus/clinc_val_with_labels.csv")
test_df = pd.read_csv("data/hugging_face/plus/clinc_test_with_labels.csv")

# ---------------------------------------------------------
# 2. Load MiniLM embedding model
# ---------------------------------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 3. Embed text
# ---------------------------------------------------------
X_train = embedder.encode(train_df["text"].tolist(), batch_size=64, convert_to_numpy=True)
X_val = embedder.encode(val_df["text"].tolist(), batch_size=64, convert_to_numpy=True)
X_test = embedder.encode(test_df["text"].tolist(), batch_size=64, convert_to_numpy=True)

y_train = train_df["intent_label"]
y_val = val_df["intent_label"]
y_test = test_df["intent_label"]

# ---------------------------------------------------------
# 4. Train Logistic Regression
# ---------------------------------------------------------
clf = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    C=3.0,
    solver="lbfgs"
)

clf.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate on validation + test
# ---------------------------------------------------------
val_preds = clf.predict(X_val)
test_preds = clf.predict(X_test)

print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))

# ---------------------------------------------------------
# 6. Save model + embedder
# ---------------------------------------------------------
joblib.dump(clf, "model/intent_logreg_minilm.pkl")
embedder.save("model/minilm_embedder")

print("Training complete. Model and embedder saved.")
