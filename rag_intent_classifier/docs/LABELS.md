
# Human-readable intent labels

This package supports 156 intents in total:

- 150 original CLINC150 intents
- 6 additional meta-intents for RAG routing:
  - follow_up
  - connect_to_human
  - escalate_issue
  - negative_sentiment
  - urgent_attention_required
  - general_customer_support

The full mapping is stored in the bundled model asset at [rag_intent_classifier/models/label_map.joblib](../models/label_map.joblib).

## Examples

```python
from rag_intent_classifier import infer_intent

infer_intent("I'm not satisfied with your last answer")
# ['follow_up']

infer_intent("please assist immediately, my card is showing unauthorized activity")
# ['urgent_attention_required']

infer_intent("i want a real person to help me from here")
# ['connect_to_human']
```

These labels are intended for routing and retrieval decisions before the main RAG search stage.
