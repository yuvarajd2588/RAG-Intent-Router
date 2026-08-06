
---

## 📄 `LABELS.md`

```markdown
# Human-Readable Intent Labels

This package supports **156 intents**:

- 150 original CLINC150 intents  
- 1 new intent: `follow_up`

Example labels:

- translate  
- find_phone  
- payday  
- application_status  
- insurance_claims  
- policy_renewal  
- account_update  
- ...
- **follow_up**

The full mapping is stored in: rag_intent_classifier/models/label_map.joblib


Use inference to see human-readable labels:

```python
infer_intent("I'm not satisfied with your last answer")
Output: ['follow_up']


