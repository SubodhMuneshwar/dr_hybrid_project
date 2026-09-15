import os
import logging
import argparse
import numpy as np
import joblib
import cv2
from . import config
from .data import advanced_preprocess_image
from .features import (
    get_deep_feature_model,
    extract_deep_features,
    extract_deep_features_torch,
    extract_deep_features_fallback,
    extract_lbp,
    extract_haralick,
    TORCH_AVAILABLE,
    TF_AVAILABLE,
)
from .explain import grad_cam

logger = logging.getLogger(__name__)

# Global memory caches for classifier and scaler to prevent multi-gigabyte disk loads on each scan
_CACHED_CLASSIFIER = None
_CACHED_SCALER = None


def _load_classifier():
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is not None:
        return _CACHED_CLASSIFIER
    candidates = [
        os.path.join(config.MODELS_DIR, "stacking_calibrated.pkl"),
        os.path.join(config.MODELS_DIR, "votingclassifier_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            _CACHED_CLASSIFIER = joblib.load(path)
            return _CACHED_CLASSIFIER
    raise FileNotFoundError(
        f"Trained model not found. Checked: {candidates}. "
        "Download from Drive (README) or run: python -m src.pipeline --train"
    )


def _load_scaler():
    global _CACHED_SCALER
    if _CACHED_SCALER is not None:
        return _CACHED_SCALER
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            _CACHED_SCALER = joblib.load(scaler_path)
            return _CACHED_SCALER
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}")
    return None


def _expected_dims():
    """Return (expected_total, expected_deep) from scaler if present, else (None, None)."""
    scaler = _load_scaler()
    if scaler is not None:
        total = int(getattr(scaler, "n_features_in_", 0) or 0)
        if total:
            # LBP 26 + Haralick 24 = 50
            deep = total - 50
            return total, deep
    return None, None


def _get_deep(extractor_name, expected_deep):
    """Pick GAP vs flattened VGG16 based on expected_deep when TF is available."""
    if not TF_AVAILABLE:
        return None, None
    gap_dims = {"vgg16": 512, "densenet121": 1024, "mobilenetv2": 1280, "inceptionresnetv2": 1536}
    if expected_deep is not None and expected_deep == gap_dims.get(extractor_name, 512):
        return get_deep_feature_model(extractor_name, use_gap=True)
    if extractor_name == "vgg16" and expected_deep is not None and expected_deep > 2000:
        return get_deep_feature_model(extractor_name, use_gap=False)
    return get_deep_feature_model(extractor_name, use_gap=True)


def infer_image(image_path):
    """
    Predict DR severity from retinal image and generate Grad-CAM saliency overlay.

    Args:
        image_path (str): Path to fundus image

    Returns:
        tuple: (prediction_class, probabilities, heatmap_path, preprocessed_path)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    clf = _load_classifier()
    target_size = getattr(config, "TARGET_SIZE", (256, 256))
    img_bgr, img_clahe = advanced_preprocess_image(image_path, target_size=target_size)

    if img_bgr is None:
        raise ValueError(f"Failed to read image at {image_path}")

    expected_total, expected_deep = _expected_dims()

    # 1. Deep features extraction (PyTorch preferred, TensorFlow secondary, Fallback tertiary)
    deep_feat = None
    if TORCH_AVAILABLE:
        try:
            deep_feat = extract_deep_features_torch(img_bgr, expected_deep=expected_deep, target_size=target_size)
        except Exception as e:
            logger.warning(f"PyTorch deep feature extraction failed: {e}")

    if deep_feat is None and TF_AVAILABLE:
        try:
            deep_model, preprocess_fn = _get_deep(config.FEATURE_EXTRACTOR_MODEL, expected_deep)
            deep_feat = extract_deep_features(img_bgr, deep_model, preprocess_fn)
            if expected_deep is not None and deep_feat.size != expected_deep:
                if deep_feat.size < expected_deep:
                    pad = np.zeros(expected_deep - deep_feat.size, dtype=deep_feat.dtype)
                    deep_feat = np.concatenate([deep_feat, pad])
                else:
                    deep_feat = deep_feat[:expected_deep]
        except Exception as e:
            logger.warning(f"TensorFlow deep feature extraction failed: {e}")

    if deep_feat is None:
        if expected_deep is None:
            expected_deep = 32768 if target_size == (256, 256) else 512
        deep_feat = extract_deep_features_fallback(img_bgr, expected_deep_dim=expected_deep)

    # 2. Handcrafted texture features: LBP (26) + Haralick GLCM (24) = 50
    lbp_feat = extract_lbp(img_clahe)
    haralick_feat = extract_haralick(img_clahe)
    fused = np.concatenate([deep_feat, lbp_feat, haralick_feat]).reshape(1, -1)

    # 3. Normalization via StandardScaler
    scaler = _load_scaler()
    if scaler is not None:
        exp = int(getattr(scaler, "n_features_in_", fused.shape[1]))
        if fused.shape[1] != exp:
            if fused.shape[1] < exp:
                pad = np.zeros((1, exp - fused.shape[1]), dtype=fused.dtype)
                fused = np.concatenate([fused, pad], axis=1)
            else:
                fused = fused[:, :exp]
        try:
            fused = scaler.transform(fused)
        except Exception as e:
            logger.warning(f"Scaler transformation warning: {e}")

    # 4. Model Prediction
    proba = clf.predict_proba(fused)[0]
    proba = np.array(proba, dtype=float)
    if proba.size != len(config.CLASS_NAMES):
        tmp = np.zeros(len(config.CLASS_NAMES))
        tmp[:min(len(tmp), len(proba))] = proba[:min(len(tmp), len(proba))]
        if tmp.sum() > 0:
            tmp = tmp / tmp.sum()
        else:
            tmp = np.ones(len(tmp)) / len(tmp)
        proba = tmp
    pred = int(np.argmax(proba))

    overlay, _ = grad_cam(img_bgr, target_size=target_size)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)

    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    out_path = os.path.join(config.OUTPUTS_DIR, f"{name}_gradcam.png")
    preprocessed_path = os.path.join(config.UPLOADS_DIR, f"{name}_input.png")
    legacy_overlay_path = os.path.join(config.OUTPUTS_DIR, "gradcam_overlay.png")

    cv2.imwrite(out_path, overlay)
    cv2.imwrite(legacy_overlay_path, overlay)
    cv2.imwrite(preprocessed_path, img_bgr)

    # Also mirror to static so the index hero always has a file
    try:
        static_overlay = os.path.join(os.path.dirname(__file__), "..", "app", "static", "gradcam_overlay.png")
        static_overlay = os.path.abspath(static_overlay)
        os.makedirs(os.path.dirname(static_overlay), exist_ok=True)
        cv2.imwrite(static_overlay, overlay)
    except Exception:
        pass

    return pred, proba, out_path, preprocessed_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()
    p, pr, hp = infer_image(args.image)
    print("Prediction:", p)
    print("Probabilities:", pr)
    print("Grad-CAM saved to:", hp)
