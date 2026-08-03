import re
from pathlib import Path
import csv
from typing import Tuple


ABBREVIATION_MAP = {
    "pls": "please",
    "plz": "please",
    "can u": "can you",
    "cant": "cannot",
    "won't": "will not",
    "don't": "do not",
    "i'm": "i am",
    "it's": "it is",
    "asap": "as soon as possible",
    "hr": "human resources",
    "faq": "frequently asked questions",
    "id": "identification",
    "dob": "date of birth",
    "claim no": "claim number",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()

    for phrase, replacement in ABBREVIATION_MAP.items():
        pattern = r"\b" + re.escape(phrase) + r"\b"
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_dataset(input_path: Path, output_path: Path) -> Tuple[int, int]:
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))

    normalized_rows = []
    for row in rows:
        normalized_query = normalize_text(row["query"])
        normalized_rows.append({"query": normalized_query, "department": row["department"]})

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["query", "department"])
        writer.writeheader()
        writer.writerows(normalized_rows)

    return len(rows), len(normalized_rows)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "data" / "raw_dataset.csv"
    output_path = repo_root / "phase1-data-normalize" / "normalized_dataset.csv"
    count, _ = normalize_dataset(input_path, output_path)
    print(f"Normalized {count} rows")
