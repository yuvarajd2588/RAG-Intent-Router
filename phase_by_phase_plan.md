# Intent Router MVP Plan

## Goal
Build a lightweight Python package that routes insurance-related user queries to the correct department intent such as claims, agency, premium renewal, HR/benefits, or unknown. The first version should output only the department label.

## Recommended approach
Use a hybrid approach:
- apply a small manual normalization layer for common shorthand and abbreviations
- train on real-world chat-style examples so the model learns informal phrasing, typos, and short queries
- add a conversation-aware routing layer so follow-up questions do not get incorrectly sent to the intent router
- combine heuristics with a small trained classifier for follow-up vs new-intent detection

## Phase 1 — Define the scope
- Keep the project focused on intent routing only
- Output one department label per query
- Start with a small set of departments:
  - claims
  - agency
  - premium_renewal
  - hr_benefits
  - unknown

## Phase 2 — Create the dataset
- Collect or create a labeled dataset of insurance-related queries
- Start with around 100–300 examples for an MVP
- Include both formal phrasing and short chat-style phrasing
- Include common typos and shorthand where possible

## Phase 3 — Build the cleaning and normalization layer
- Lowercase the text
- Trim whitespace and normalize punctuation
- Apply a small manual dictionary for common abbreviations and shorthand
- Examples:
  - ASAP → as soon as possible
  - HR → human resources
  - FAQ → frequently asked questions
- Standardize obvious insurance domain terms where helpful
- Keep the normalization light and consistent

## Phase 4 — Train the intent router
- Use a sentence embedding model such as `all-MiniLM-L6-v2`
- Train a lightweight classifier such as logistic regression or linear SVM
- Save the trained model and label mapping
- Add an `unknown` fallback for low-confidence predictions

## Phase 5 — Build inference
- Create a simple Python API for prediction
- Create a small command-line interface for quick testing
- Support both single-turn routing and multi-turn conversation context
- Use a hybrid decision layer to classify whether the input is:
  - a new department intent
  - a follow-up clarification
  - a conversation continuation
- Output:
  - input query
  - predicted department
  - confidence score
- Avoid routing follow-up questions such as “I don’t understand the last answer” to the department store when the conversation context suggests they should stay within the chat flow

## Phase 6 — Package for pip install
- Package the code so users can install it with pip
- Publish the package to PyPI
- Users should be able to run:

```bash
pip install intent-router
```

## Phase 7 — Host the model artifact
- Upload the trained model artifact to Hugging Face
- Make it easy for users to download and load the model locally
- This supports reproducibility and simple deployment

## Phase 8 — Create a demo and results summary
- Build a simple demo script or notebook
- Show example queries and predicted departments
- Record the results clearly for a LinkedIn post or project README
- pip-installable Python package
- public model hosted on Hugging Face
