from rag_intent_classifier import infer_intent, list_available_models


def test_package_import_and_inference():
    models = list_available_models()
    assert models, "expected packaged models to be available"

    prediction = infer_intent("How do I renew my policy?")
    assert len(prediction) == 1
    assert isinstance(prediction[0], str)
    assert prediction[0]
