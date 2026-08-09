# rag_intent_classifier

A lightweight, production-ready, offline intent classifier for RAG systems. It sits before retrieval and routes user queries into the right workflow early.

## Why this package matters

Most RAG systems treat retrieval as the main decision point, but intent classification is often the missing first step. This package adds a reliable layer that detects meta-intents such as:

- follow_up
- connect_to_human
- escalate_issue
- negative_sentiment
- urgent_attention_required
- general_customer_support

These labels help reduce irrelevant retrieval and improve routing quality in support, customer-service, and knowledge-base applications.

## What the package provides

- 156 intents total (150 CLINC + 6 additional meta-intents)
- MiniLM-based embeddings
- Multiple packaged classical ML classifiers
- Human-readable labels for downstream logic
- Fully offline inference
- Simple Python API and CLI

## Installation

Python 3.9 to 3.12 is the currently supported range.

```bash
pip install rag_intent_classifier
```

## Quick usage

```python
from rag_intent_classifier import infer_intent

result = infer_intent("How do I renew my policy?")
print(result)
```

Example output:
```python
[{"intent": "general_customer_support", "confidence": 0.91, "reason": "predicted by classifier with probability-based score"}]
```

Use a confidence threshold such as 0.60 to 0.85 to decide when to fall back to an LLM, human handoff, or custom business logic. A threshold around 0.60 is a practical default for many routing workflows, while more conservative systems may prefer 0.70 or higher.

## RAG placement

```mermaid
flowchart TD
    A[👤 User query] --> B[🧠 Intent classifier]
    B --> C{Intent type}
    C -->|Meta-intent| D[🧭 Route to appropriate workflow]
    C -->|Domain intent| E[🔎 Retrieval stage]
    D --> F[⚡ Faster, more focused search]
    E --> G[🗂️ Vector DB + BM25]
    F --> H[⚖️ RRF fusion]
    G --> H
    H --> I[🎯 Optional cross-encoder rerank]
    I --> J[🤖 LLM answer generation]
```

This helps shrink the search space and route the request to the right handling path before the main retrieval step.

## Sample inputs and outputs

Here are a few example predictions the model can return:

```python
from rag_intent_classifier import infer_intent

print(infer_intent("Can you connect me to a human?"))
print(infer_intent("Can you follow up on my previous request?"))
print(infer_intent("I'm extremely unhappy with this service."))
```

Example output:

```python
[{'intent': 'connect_to_human', 'confidence': 0.95, 'reason': 'predicted by classifier with probability-based score'}]
[{'intent': 'follow_up', 'confidence': 0.91, 'reason': 'predicted by classifier with probability-based score'}]
[{'intent': 'negative_sentiment', 'confidence': 0.97, 'reason': 'predicted by classifier with probability-based score'}]
```
