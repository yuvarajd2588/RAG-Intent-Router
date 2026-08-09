from rag_intent_classifier import infer_intent, list_available_models


def test_package_import_and_inference():
    models = list_available_models()
    assert models, "expected packaged models to be available"

    prediction = infer_intent("How do I renew my policy?")
    print(prediction)
    assert len(prediction) == 1
    assert isinstance(prediction[0], dict)
    assert prediction[0]["intent"]
    assert "confidence" in prediction[0]
    assert "reason" in prediction[0]


def test_new_custom_labels_are_returned_as_names():
    prediction = infer_intent("connect me to a human")
    print(prediction)
    assert len(prediction) == 1
    assert prediction[0]["intent"] == "connect_to_human"
