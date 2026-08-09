# Usage examples

## Basic inference

```python
from rag_intent_classifier import infer_intent

infer_intent("How do I renew my policy?")
```

## Using a different classifier

```python
from rag_intent_classifier import infer_intent

result = infer_intent("How do I renew my policy?", model="GaussianNB")
print(result)
```

## Batch inference

```python
from rag_intent_classifier import infer_intent

result = infer_intent([
    "How do I renew my policy?",
    "I'm not satisfied with your last answer",
])
print(result)
```

## List available packaged models

```python
from rag_intent_classifier import list_available_models

print(list_available_models())
```


## Quick usage

```python
from rag_intent_classifier import infer_intent

result = infer_intent("How do I renew my policy?")
print(result)
```

Each result includes an `intent`, a `confidence` score when available, and a `reason` string. A practical starting point is to use a confidence threshold such as `0.60` to `0.85` before falling back to an LLM or custom routing logic. A threshold around `0.60` is a practical default for many routing workflows, while more conservative systems may prefer `0.70` or higher.

## CLI usage

```bash
`rag_intent_classifier "How do I renew my policy?"`
```