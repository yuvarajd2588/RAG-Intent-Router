# Available Classifier Models

This package includes several trained classifier models for intent prediction:

- LogisticRegression (default)
- GaussianNB
- LinearSVC
- PassiveAggressiveClassifier
- SGDClassifier
- RidgeClassifier

Logistic Regression is chosen as the default because it performed strongly on the validation split for this training data and is a reliable general-purpose choice for predicting intents.

Users are also free to select any of the other provided models when calling inference.

## Validation Summary

| Model | Validation Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| LogisticRegression | 0.9453 | 0.9480 | 0.9447 |
| LinearSVC | 0.9464 | 0.9503 | 0.9445 |
| GaussianNB | 0.8985 | 0.9087 | 0.9020 |
| PassiveAggressive | 0.9388 | 0.9425 | 0.9364 |
| SGDClassifier | 0.9317 | 0.9355 | 0.9281 |
| RidgeClassifier | 0.8808 | 0.8831 | 0.8710 |

## Training Data

All models are trained using:

- MiniLM‑L6‑v2 embeddings
- 15k CLINC150 dataset
- + 400 follow‑up intent samples
- + 400 negative sentiment samples
- + 400 escalate issue samples
- + 400 urgent attention required samples
- + 400 general customer support samples
- + 400 connect to human samples

## Inference Example

```python
from rag_intent_classifier import infer_intent

infer_intent("How do I renew my policy?")
infer_intent("How do I renew my policy?", model="LinearSVC")
