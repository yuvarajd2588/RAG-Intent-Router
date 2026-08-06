
# rag-intent-classifier

A lightweight MiniLM-based intent classifier for RAG systems. It is designed to handle both domain intents and follow-up/meta-intents such as clarification or revision requests.

## Why this project exists

Many RAG systems skip intent classification entirely, and even when they include a classifier, they often miss follow-up requests such as:

- "I'm not satisfied with your last answer"
- "Give me the last answer in 3 bullet points"
- "Can you clarify your previous response?"

Those are not domain intents; they are meta-intents that should be routed separately to include chat history and skip the chunk search. This package addresses that gap.

## Included capabilities

- 151 intents (150 CLINC + 1 follow-up intent)
- MiniLM embeddings
- Multiple packaged classifier models
- Offline inference with no external API dependency
- Simple Python API and CLI

## Installation

```bash
pip install rag-intent-classifier
```

## Quick start

```python
from rag_intent_classifier import infer_intent

print(infer_intent("How do I renew my policy?"))
```

## CLI usage

```bash
rag-intent-classifier "How do I renew my policy?"
```

## Development install

```bash
git clone https://github.com/yuvarajd2588/rag-intent-classifier.git
cd rag-intent-classifier
pip install -e .
```

## Packaging check

```bash
python -m build
python -m twine check dist/*
```
