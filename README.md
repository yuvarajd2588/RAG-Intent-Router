
# rag_intent_classifier

A lightweight MiniLM-based intent classifier for RAG systems. It is designed to handle both domain intents and follow-up, negative sentiment & escalation intents such as clarification or revision requests.

## Why this project exists

Many RAG systems skip intent classification entirely, and even when they include a classifier, they often miss follow-up, negativ sentiments, escalation, urgent issue requests such as:


| Query | Label |
|-------|--------|
| I'm not satisfied with your last answer | follow-up |
| Give me the last answer in 3 bullet points | follow-up |
| Can you clarify your previous response? | follow-up |
| please assist immediately, my card is showing unauthorized activity | urgent_attention_required |
| i want a real person to help me from here | connect_to_human |
| this needs to be escalated to someone more capable | escalate_issue |
| where can i adjust my profile visibility settings | general_customer_support |
| this whole process feels extremely frustrating | negative_sentiment |


Those are not domain intents; they are meta-intents that should be routed separately to include chat history or to the right department or skip the doc chunk search. This package addresses that gap.

## Included capabilities

- 156 intents (150 CLINC + 6 additional intents)
- MiniLM embeddings
- Multiple packaged classifier models
- Offline inference with no external API dependency
- Simple Python API and CLI

## Installation

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
