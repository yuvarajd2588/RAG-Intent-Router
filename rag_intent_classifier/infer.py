import argparse
import os
import joblib

PACKAGE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PACKAGE_DIR, "models")

_encoder = None
_label_map = None


def _patch_transformer_model_compatibility():
    """Make older serialized BERT-based encoder assets usable with newer transformers versions."""
    try:
        from transformers.models.bert.modeling_bert import BertSdpaSelfAttention, BertSelfAttention
    except Exception:
        return

    for cls in [BertSdpaSelfAttention, BertSelfAttention]:
        if not hasattr(cls, "require_contiguous_qkv"):
            setattr(cls, "require_contiguous_qkv", False)


def _patch_tokenizer_compatibility(encoder):
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        return

    special_tokens_map = getattr(tokenizer, "_special_tokens_map", None)
    if not special_tokens_map:
        special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}

    for public_name, private_name in [
        ("pad_token", "_pad_token"),
        ("bos_token", "_bos_token"),
        ("eos_token", "_eos_token"),
        ("unk_token", "_unk_token"),
        ("sep_token", "_sep_token"),
        ("cls_token", "_cls_token"),
        ("mask_token", "_mask_token"),
        ("additional_special_tokens", "_additional_special_tokens"),
    ]:
        if not hasattr(tokenizer, private_name):
            value = special_tokens_map.get(public_name)
            if value is None:
                value = getattr(tokenizer, public_name, None)
            if value is not None:
                setattr(tokenizer, private_name, value)
                if public_name == "pad_token":
                    try:
                        tokenizer._pad_token_id = tokenizer.convert_tokens_to_ids(value)
                    except Exception:
                        pass

    if not hasattr(tokenizer, "_additional_special_tokens"):
        setattr(tokenizer, "_additional_special_tokens", special_tokens_map.get("additional_special_tokens", []))

    transformer_module = None
    if hasattr(encoder, "__getitem__"):
        try:
            transformer_module = encoder[0]
        except Exception:
            transformer_module = None

    if transformer_module is not None:
        auto_model = getattr(transformer_module, "auto_model", None)
        if auto_model is not None:
            config_obj = getattr(auto_model, "config", None)
            if config_obj is not None:
                for attr_name in ["output_attentions", "output_hidden_states", "return_dict", "use_cache"]:
                    if not hasattr(config_obj, attr_name):
                        setattr(config_obj, attr_name, False if attr_name in {"output_attentions", "output_hidden_states"} else True)


def _load_assets():
    global _encoder, _label_map
    if _encoder is None or _label_map is None:
        _patch_transformer_model_compatibility()
        _encoder = joblib.load(os.path.join(MODEL_DIR, "minilm_encoder.joblib"))
        _patch_tokenizer_compatibility(_encoder)
        _label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.joblib"))
    return _encoder, _label_map


def infer_intent(text, model="LogisticRegression"):
    """
    Predict intent using the specified model.
    Default model = LogisticRegression.
    """

    encoder, label_map = _load_assets()
    model_path = os.path.join(MODEL_DIR, f"{model}.joblib")
    if not os.path.exists(model_path):
        raise ValueError(
            f"Model '{model}' not found. Available models: {list_available_models()}"
        )

    clf = joblib.load(model_path)

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    embeddings = encoder.encode(texts, show_progress_bar=False)
    numeric_preds = clf.predict(embeddings)
    human_preds = [label_map[p] for p in numeric_preds]

    return human_preds


def list_available_models():
    excluded = {"label_map", "minilm_encoder"}
    return [
        f.replace(".joblib", "")
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".joblib") and f.replace(".joblib", "") not in excluded
    ]


def main():
    parser = argparse.ArgumentParser(description="Classify the intent of one or more queries")
    parser.add_argument("text", nargs="+", help="Query text to classify")
    parser.add_argument("--model", default="LogisticRegression", help="Model to use")
    args = parser.parse_args()

    predictions = infer_intent(args.text, model=args.model)
    for prediction in predictions:
        print(prediction)


if __name__ == "__main__":
    main()
