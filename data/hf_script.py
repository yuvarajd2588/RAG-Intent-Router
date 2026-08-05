from datasets import load_dataset
import pandas as pd
import json

# ---------------------------------------------------------
# 1. Load CLINC150 (small version)
# ---------------------------------------------------------
ds = load_dataset("clinc_oos", "plus")
train = ds["train"]
test = ds["test"]
val = ds["validation"]

# ---------------------------------------------------------
# 2. Extract intent label names from metadata
# ---------------------------------------------------------
intent_labels = train.features["intent"].names

# ---------------------------------------------------------
# 3. Convert train/test/val splits to DataFrames
# ---------------------------------------------------------
train_df = train.to_pandas()
test_df = test.to_pandas()
val_df = val.to_pandas()

# ---------------------------------------------------------
# 4. Add readable intent labels
# ---------------------------------------------------------
train_df["intent_label"] = train_df["intent"].apply(lambda x: intent_labels[x])
test_df["intent_label"] = test_df["intent"].apply(lambda x: intent_labels[x])
val_df["intent_label"] = val_df["intent"].apply(lambda x: intent_labels[x])

# ---------------------------------------------------------
# 5. Save updated CSV files
# ---------------------------------------------------------
train_df.to_csv("clinc_train_with_labels.csv", index=False)
test_df.to_csv("clinc_test_with_labels.csv", index=False)
val_df.to_csv("clinc_val_with_labels.csv", index=False)

# ---------------------------------------------------------
# 6. Compute label counts
# ---------------------------------------------------------
label_counts = train_df["intent_label"].value_counts().sort_index()

# ---------------------------------------------------------
# 7. Build JSON summary
# ---------------------------------------------------------
summary = {
    "total_labels": len(intent_labels),
    "labels": intent_labels,
    "label_counts": label_counts.to_dict(),
    "total_train_samples": len(train_df),
    "total_test_samples": len(test_df),
    "total_validation_samples": len(val_df),
}

with open("clinc_intent_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("CLINC150 dataset downloaded, processed, and summarized.")

if __name__ == "__main__":
    print("Banking77 dataset downloaded and saved as CSV files.")
