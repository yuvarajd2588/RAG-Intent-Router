import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
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
clf = make_pipeline(
    StandardScaler(with_mean=False),
    LinearSVC(C=3.0, max_iter=8000)
)

# clf = CalibratedClassifierCV(clf)
clf.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate on validation + test
# ---------------------------------------------------------
# print("calling predict_probablity on test set to get probabilities for each class")
# clf.predict_proba(X_test)

val_preds = clf.predict(X_val)
test_preds = clf.predict(X_test)

print("Linear SVC Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Linear SVC Test Accuracy:", accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))

# ---------------------------------------------------------
# 6. Save model + embedder
# ---------------------------------------------------------
joblib.dump(clf, "model/intent_svc_minilm.pkl")
embedder.save("model/minilm_embedder")

print("Training complete. Model and embedder saved.")
