import argparse
import os
import warnings

import joblib

warnings.filterwarnings("ignore", message=r"`resume_download` is deprecated.*", category=FutureWarning)

from sentence_transformers import SentenceTransformer

PACKAGE_DIR = os.path.dirname(__file__)
PACKAGE_MODEL_DIR = os.path.join(PACKAGE_DIR, "models")
CACHE_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "rag_intent_classifier", "models")


def _get_model_dir():
    if os.path.isdir(PACKAGE_MODEL_DIR):
        return PACKAGE_MODEL_DIR
    os.makedirs(CACHE_MODEL_DIR, exist_ok=True)
    return CACHE_MODEL_DIR


MODEL_DIR = _get_model_dir()

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


def _load_encoder():
    encoder_path = os.path.join(MODEL_DIR, "minilm_encoder.joblib")
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    else:
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    _patch_tokenizer_compatibility(encoder)
    return encoder


def _load_assets():
    global _encoder, _label_map
    if _encoder is None or _label_map is None:
        _patch_transformer_model_compatibility()
        _encoder = _load_encoder()

        label_map_path = os.path.join(MODEL_DIR, "label_map.joblib")
        if os.path.exists(label_map_path):
            _label_map = joblib.load(label_map_path)
        else:
            raise RuntimeError("Label map is not available. Reinstall the package with the bundled model assets or train them locally.")
    return _encoder, _label_map


def _resolve_label_name(prediction, label_map):
    if label_map is None:
        return str(prediction)

    prediction_str = str(prediction)

    if prediction_str in label_map:
        mapped_value = label_map[prediction_str]
        if isinstance(mapped_value, str):
            if mapped_value.isdigit() or (mapped_value.startswith("-") and mapped_value[1:].isdigit()):
                return prediction_str
            return mapped_value
        return str(mapped_value)

    for key, value in label_map.items():
        if str(key) == prediction_str:
            return str(key)
        if str(value) == prediction_str:
            return str(key)

    return prediction_str


def infer_intent(text, model="LogisticRegression"):
    """
    Predict intent using the specified model.
    Default model = LogisticRegression.
    """

    encoder, label_map = _load_assets()
    model_path = os.path.join(MODEL_DIR, f"{model}.joblib")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model '{model}' is not available in {MODEL_DIR}")

    clf = joblib.load(model_path)

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    embeddings = encoder.encode(texts, show_progress_bar=False)
    numeric_preds = clf.predict(embeddings)
    human_preds = [_resolve_label_name(prediction, label_map) for prediction in numeric_preds]

    return human_preds


def list_available_models():
    excluded = {"label_map", "minilm_encoder", "KNN"}
    if not os.path.isdir(MODEL_DIR):
        return []
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
