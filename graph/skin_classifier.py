"""CNN-based skin type and acne classifier wrapping two EfficientNet-B0 models."""

import io
import os

# Force legacy Keras — these SavedModels were produced with TF 2.x Keras 2
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIN_MODEL_PATH = os.path.join(_ROOT, "models", "skin_cnn")
ACNE_MODEL_PATH = os.path.join(_ROOT, "models", "acne_cnn")

SKIN_LABELS = ["dry", "normal", "oily"]
ACNE_LABELS = ["low", "moderate", "severe"]

_skin_model = None
_acne_model = None
_load_error = None


def _load_models():
    """Lazy single-load of both Keras models. Returns (skin, acne) or (None, None)."""
    global _skin_model, _acne_model, _load_error
    if _skin_model is not None and _acne_model is not None:
        return _skin_model, _acne_model
    if _load_error is not None:
        return None, None
    try:
        import tensorflow as tf
        _skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH, compile=False)
        _acne_model = tf.keras.models.load_model(ACNE_MODEL_PATH, compile=False)
        return _skin_model, _acne_model
    except Exception as e:
        _load_error = str(e)
        print(f"[skin_classifier] Failed to load TF models: {e}")
        return None, None


def _preprocess(image_bytes: bytes):
    """Decode image bytes -> 1x224x224x3 float32 tensor."""
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype="float32")
    return arr[None, :, :, :]


def classify(image_bytes: bytes) -> dict:
    """
    Returns skin type/acne predictions with probabilities, or
    {"available": False, "error": <reason>} on failure.
    """
    skin_m, acne_m = _load_models()
    if skin_m is None or acne_m is None:
        return {"available": False, "error": _load_error or "models not loaded"}

    try:
        x = _preprocess(image_bytes)
    except Exception as e:
        return {"available": False, "error": f"image decode failed: {e}"}

    try:
        skin_probs = skin_m.predict(x, verbose=0)[0]
        acne_probs = acne_m.predict(x, verbose=0)[0]
    except Exception as e:
        return {"available": False, "error": f"inference failed: {e}"}

    skin_probs = _softmax_if_needed(skin_probs)
    acne_probs = _softmax_if_needed(acne_probs)

    skin_idx = int(skin_probs.argmax())
    acne_idx = int(acne_probs.argmax())

    return {
        "available":       True,
        "skin_type":       SKIN_LABELS[skin_idx],
        "skin_confidence": float(skin_probs[skin_idx]),
        "skin_probs":      {l: float(p) for l, p in zip(SKIN_LABELS, skin_probs)},
        "acne":            ACNE_LABELS[acne_idx],
        "acne_confidence": float(acne_probs[acne_idx]),
        "acne_probs":      {l: float(p) for l, p in zip(ACNE_LABELS, acne_probs)},
    }


def _softmax_if_needed(v):
    import numpy as np
    v = np.asarray(v, dtype="float32")
    s = v.sum()
    if 0.99 <= s <= 1.01 and (v >= 0).all():
        return v
    e = np.exp(v - v.max())
    return e / e.sum()
