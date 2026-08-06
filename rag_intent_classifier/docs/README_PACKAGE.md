# rag-intent-classifier

A lightweight, production-ready **intent classifier** built using MiniLM embeddings and classical ML models.

## Why this package exists

Most RAG architecture diagrams completely miss one critical component:

### **Intent Classification — especially Follow-Up Intent Detection**

This package provides:

- 150 original CLINC intents  
- **+ 1 new critical intent: `follow_up`**  
- Total: **151 intents**

Follow-up queries like:

- "I'm not satisfied with your last answer"
- "Give me the last answer in 3 bullet points"
- "Can you clarify your previous response?"

are **not domain intents** — they are **meta-intents** that must be routed differently in RAG systems.

This classifier solves that missing piece.

---

## Features

- MiniLM-L6-v2 embeddings  
- Multiple classifier models packaged:
  - LogisticRegression (default)
  - GaussianNB
  - LinearSVC
  - PassiveAggressive
  - SGDClassifier
  - RidgeClassifier
  - KNN
- Human-readable intent labels  
- Follow-up intent classification  
- Zero external downloads required  
- Fully offline inference  
- pip-installable  

---

## Installation

```bash
pip install rag-intent-classifier
