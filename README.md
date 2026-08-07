
# rag_intent_classifier

A production-ready, offline intent classifier that can sit right before any RAG system, domain-aware router, or any routing layer where fast intent decisions are needed. It runs before retrieval or downstream routing so your application can detect follow-up, escalation, urgent, human-handoff, negative-sentiment, and general-support requests early and route them intelligently.

## Why this package exists

Most RAG pipelines, domain-aware routing systems, and other decision-heavy workflows focus on retrieval or downstream handling, but they often skip the most important first step: User intent classification.

This package helps you identify whether a user query is:
- a normal domain request,
- a follow-up question,
- a request to connect to a human,
- an escalation,
- an urgent issue,
- or negative sentiment that should not trigger normal search.
- 150 default labels (The full mapping is stored in the bundled model asset at *rag_intent_classifier/models/label_map.joblib*)

That makes retrieval more accurate and reduces unnecessary document search.

## What it does

- Classifies intents using MiniLM embeddings and packaged classical ML models
- Returns human-readable labels for downstream routing
- Works offline with no external API dependency
- Supports a simple Python API and CLI

## Supported meta-intent labels

The current release surfaces these labels for routing-heavy RAG scenarios:

- follow_up
- connect_to_human
- escalate_issue
- negative_sentiment
- urgent_attention_required
- general_customer_support
- and additional 150 intents

These are especially useful for routing to a human, skipping standard retrieval, when to send chat history, narrowing search to the right support context, etc.

## Quick comparison

| Without using intent classifier | With using intent classifier |
| --- | --- |
| Every query goes through the same retrieval flow | Urgent, follow-up, and escalation requests can be detected early |
| More irrelevant context may be searched | Routing becomes more precise and efficient |
| Human handoff needs are harder to identify | Support workflows can be triggered faster and more accurately |

This model is trained on a total of 17k+ labelled rows, with 3.8k+ validation rows and 5.7k+ test rows to support reliable intent routing.

## Architecture in a RAG pipeline

The package acts as the first decision layer in a RAG system.

### Intent Classifier architecture
![Architecture Diagram](https://raw.githubusercontent.com/yuvarajd2588/rag_Intent_classifier/main/Intent_classifier_arch.png)

In practice, once the intent is classified, the retrieval step becomes smaller and more precise: the system can route to the right workflow, focus on the most relevant evidence and reduce the search area/scope dramatically. 

## Included capabilities

- 156 intents total (150 CLINC + 6 additional meta-intents)
- MiniLM embeddings
- Multiple packaged classifier models
- Offline inference
- Human-readable labels
- Simple Python API and CLI

## Installation

Python 3.9 to 3.12 is the currently supported range for this package. This release uses a pinned dependency stack for reproducible installs and model compatibility. Python 3.13 is not currently targeted for this release.

```bash
pip install rag_intent_classifier
```

## Quick start

```python
from rag_intent_classifier import infer_intent

print(infer_intent("How do I renew my policy?"))
```

## CLI usage

```bash
rag_intent_classifier "How do I renew my policy?"
```

## Development install

```bash
git clone https://github.com/yuvarajd2588/rag_intent_classifier.git
cd rag_intent_classifier
pip install -e .
```
