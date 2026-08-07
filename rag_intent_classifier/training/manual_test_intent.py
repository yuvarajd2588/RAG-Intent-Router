# file: inference_intent.py

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_intent_classifier.infer import infer_intent


def predict_intent(texts):
    return infer_intent(texts, model="LogisticRegression")


if __name__ == "__main__":
    # samples = [
    #     "I want to check my claim status",
    #     "How do I renew my policy?",
    #     "Can I update my address?",
    #     "how would you say fly in italian",
    #     "I'm not satisfied with last answer",
    #     "can you give last answer in 3 bullet points",
    #     "I don't like last answer",
    #     "can you continue the explanation with more formatting",
    #     "can you continue the explanation with more polish",
    #     "can you continue the explanation with more refinement",
    #     "alter the previous answer",
    #     "modify the previous answer",
    #     "elaborate on the previous answer",
    #     "can you connect me to agent"
    #     "connect me to human agent",
    #     "I want to speak to a human agent",
    #     "I'm not satisfied with the answer, connect me to a human agent",
    #     "I''m not satisfied with the answer",
    #     "Please connect me to the agent",
    #     "I don't feel good about the answer you generated",
    # ]

    samples = [
        "i'd prefer a human to help me with this",
        "can you switch me over to a human agent",
        "i need a real person to continue this conversation",
        "please hand this chat off to a human",
        "i want a human to take over from here",
        "i need a human to continue this conversation",
        "please route me to a human assistant now",
        "i want a real person to help me from here",
        "can you move this chat to a human representative",
        "i prefer speaking with a human instead of the bot",
        "this needs to be escalated to someone more capable",
        "please escalate this matter to your senior support team",
        "i want this issue pushed to a higher authority",
        "send this case to advanced support immediately",
        "i need this escalated beyond standard assistance",
        "i'm not pleased with how this is going at all",
        "this whole process feels extremely frustrating",
        "i'm annoyed because nothing seems straightforward",
        "this experience has been pretty disappointing so far",
        "i'm unhappy with the results i'm getting right now",
        "i need urgent help, my account suddenly stopped responding",
        "please assist immediately, my card is showing unauthorized activity",
        "urgent issue, my account just locked itself",
        "i need immediate support, my card was declined without reason",
        "urgent attention required, my account shows a new suspicious login",
        "how do i update my preferred contact method",
        "i need help locating my saved billing receipts",
        "where can i adjust my profile visibility settings",
        "how do i review my connected service accounts",
        "i need assistance finding my account customization menu",
]

    for prediction in predict_intent(samples):
        print(prediction)
