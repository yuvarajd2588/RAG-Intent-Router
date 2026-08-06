# Available Classifier Models

This package includes multiple trained classifier models:

- LogisticRegression (default)
- GaussianNB
- LinearSVC
- PassiveAggressiveClassifier
- SGDClassifier
- RidgeClassifier
- KNN

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
infer_intent("text", model="LinearSVC")
