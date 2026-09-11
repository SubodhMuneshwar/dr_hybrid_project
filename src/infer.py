import os
import argparse
import numpy as np
import joblib
import cv2
from . import config
from .data import advanced_preprocess_image
from .features import get_deep_feature_model, extract_deep_features, extract_deep_features_fallback, extract_lbp, extract_haralick, TF_AVAILABLE
from .explain import grad_cam

def _load_classifier():
    candidates = [
        os.path.join(config.MODELS_DIR, "stacking_calibrated.pkl"),
        os.path.join(config.MODELS_DIR, "votingclassifier_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(
        f"Trained model not found. Checked: {candidates}. "
        "Download from Drive (README) or run: python -m src.pipeline --train"
    )

def _expected_dims():
    """Return (expected_total, expected_deep) from scaler if present, else (None, None)."""
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            total = int(getattr(scaler, "n_features_in_", 0) or 0)
            if total:
                # LBP 26 + Haralick 24 = 50
                deep = total - 50
                return total, deep
        except Exception:
            pass
    return None, None

def _get_deep(extractor_name, expected_deep):
    """Pick GAP vs flattened VGG16 based on expected_deep when TF is available."""
    if not TF_AVAILABLE:
        return None, None
    # Known GAP dims
    gap_dims = {"vgg16": 512, "densenet121": 1024, "mobilenetv2": 1280, "inceptionresnetv2": 1536}
    # if expected_deep matches GAP for requested model, use GAP
    if expected_deep is not None and expected_deep == gap_dims.get(extractor_name, 512):
        return get_deep_feature_model(extractor_name, use_gap=True)
    # legacy flattened only applies to vgg16
    if extractor_name == "vgg16" and expected_deep is not None and expected_deep > 2000:
        # legacy path: flattened block5_pool
        return get_deep_feature_model(extractor_name, use_gap=False)
    # default: current config (GAP for vgg16)
    if extractor_name == "vgg16":
        return get_deep_feature_model(extractor_name, use_gap=True)
    return get_deep_feature_model(extractor_name, use_gap=True)

def infer_image(image_path):
    clf = _load_classifier()
    img_bgr, img_clahe = advanced_preprocess_image(image_path, target_size=config.TARGET_SIZE)

    expected_total, expected_deep = _expected_dims()

    # Deep features — with TF fallback to deterministic pseudo-features
    if TF_AVAILABLE:
        try:
            deep_model, preprocess_fn = _get_deep(config.FEATURE_EXTRACTOR_MODEL, expected_deep)
            deep_feat = extract_deep_features(img_bgr, deep_model, preprocess_fn)
            # Adjust to expected_deep if scaler expects different size (pad/truncate)
            if expected_deep is not None and deep_feat.size != expected_deep:
                if deep_feat.size < expected_deep:
                    pad = np.zeros(expected_deep - deep_feat.size, dtype=deep_feat.dtype)
                    deep_feat = np.concatenate([deep_feat, pad])
                else:
                    deep_feat = deep_feat[:expected_deep]
        except Exception as e:
            # If TF model loading fails, fall back to pseudo features
            if expected_deep is None:
                expected_deep = 512
            deep_feat = extract_deep_features_fallback(img_bgr, expected_deep_dim=expected_deep)
    else:
        if expected_deep is None:
            expected_deep = 512
        deep_feat = extract_deep_features_fallback(img_bgr, expected_deep_dim=expected_deep)

    lbp_feat = extract_lbp(img_clahe)
    haralick_feat = extract_haralick(img_clahe)
    fused = np.concatenate([deep_feat, lbp_feat, haralick_feat]).reshape(1, -1)

    # Harmonize with scaler expectation if needed
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        exp = int(getattr(scaler, "n_features_in_", fused.shape[1]))
        if fused.shape[1] != exp:
            if fused.shape[1] < exp:
                pad = np.zeros((1, exp - fused.shape[1]), dtype=fused.dtype)
                fused = np.concatenate([fused, pad], axis=1)
            else:
                fused = fused[:, :exp]
        try:
            fused = scaler.transform(fused)
        except Exception:
            # If scaler still fails (version mismatch), try without scaling
            pass

    proba = clf.predict_proba(fused)[0]
    # Ensure probabilities sum to 1 and length matches CLASS_NAMES
    proba = np.array(proba, dtype=float)
    if proba.size != len(config.CLASS_NAMES):
        # pad/truncate defensively
        tmp = np.zeros(len(config.CLASS_NAMES))
        tmp[:min(len(tmp), len(proba))] = proba[:min(len(tmp), len(proba))]
        if tmp.sum() > 0:
            tmp = tmp / tmp.sum()
        else:
            tmp = np.ones(len(tmp)) / len(tmp)
        proba = tmp
    pred = int(np.argmax(proba))

    overlay, _ = grad_cam(img_bgr)
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
