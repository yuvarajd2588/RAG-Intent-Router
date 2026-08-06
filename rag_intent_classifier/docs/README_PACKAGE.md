# rag_intent_classifier

A lightweight, production-ready, ML offline **intent classifier** built using MiniLM embeddings and classical ML models.

## Why this package exists

Most RAG architecture diagrams completely miss one critical component:

### **Intent Classification — especially Follow-Up Intent Detection**

This package provides:

- 150 original CLINC150 intents  
- **+ 6 new meta-intents critical for RAG systems:**
  - follow_up  
  - connect_to_human  
  - escalate_issue  
  - negative_sentiment  
  - urgent_attention_required  
  - general_customer_support  
- **Total: 156 intents**

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
- Human-readable intent labels  
- Follow-up intent classification  
- Negative sentiment detection  
- Escalation intent detection  
- Urgent attention detection  
- Connect-to-human routing  
- General customer support intent detection  
- Zero external downloads required  
- Fully offline inference  
- pip-installable  

---

## Installation

```bash
pip install rag_intent_classifier
