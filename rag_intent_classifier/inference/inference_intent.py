# file: inference_intent.py

from rag_intent_classifier.infer import infer_intent


def predict_intent(texts):
    return infer_intent(texts, model="LogisticRegression")


if __name__ == "__main__":
    samples = [
        "I want to check my claim status",
        "How do I renew my policy?",
        "Can I update my address?",
        "how would you say fly in italian",
        "I'm not satisfied with last answer",
        "can you give last answer in 3 bullet points",
        "I don't like last answer",
        "can you continue the explanation with more formatting",
        "can you continue the explanation with more polish",
        "can you continue the explanation with more refinement",
        "alter the previous answer",
        "modify the previous answer",
        "elaborate on the previous answer"
    ]

    for prediction in predict_intent(samples):
        print(prediction)
