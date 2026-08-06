
---

## 📄 `MODELS_INFO.md`

```markdown
# Available Classifier Models

This package includes multiple trained classifier models:

- LogisticRegression (default)
- GaussianNB
- LinearSVC
- PassiveAggressive
- SGDClassifier
- RidgeClassifier
- KNN

All models are trained on:

- MiniLM-L6-v2 embeddings  
- 15k CLINC150 dataset  
- + 400 synthetic follow-up intent samples  

Choose any model during inference:

```python
infer_intent("text", model="LinearSVC")