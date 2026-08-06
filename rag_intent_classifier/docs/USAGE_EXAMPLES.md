# Usage examples

## Basic inference

```python
from rag_intent_classifier import infer_intent

infer_intent("How do I renew my policy?")
```

## Using a different classifier

```python
from rag_intent_classifier import infer_intent

infer_intent("How do I renew my policy?", model="GaussianNB")
```

## Batch inference

```python
from rag_intent_classifier import infer_intent

infer_intent([
    "How do I renew my policy?",
    "I'm not satisfied with your last answer",
])
```

## List available packaged models

```python
from rag_intent_classifier import list_available_models

print(list_available_models())
```


## Quick usage

```python
from rag_intent_classifier import infer_intent

infer_intent("How do I renew my policy?")
```

## CLI usage

```bash
rag_intent_classifier "How do I renew my policy?"
```