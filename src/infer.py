import os
import argparse
import numpy as np
import joblib
import cv2
from . import config
from .data import advanced_preprocess_image
from .features import get_deep_feature_model, extract_deep_features, extract_lbp, extract_haralick
from .explain import grad_cam

def _load_classifier():
    path = os.path.join(config.MODELS_DIR, "votingclassifier_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Trained model not found.")
    return joblib.load(path)

def infer_image(image_path):
    clf = _load_classifier()
    img_bgr, img_clahe = advanced_preprocess_image(image_path, target_size=(256, 256))
    deep_model, preprocess_fn = get_deep_feature_model(config.FEATURE_EXTRACTOR_MODEL)

    deep_feat = extract_deep_features(img_bgr, deep_model, preprocess_fn)
    lbp_feat = extract_lbp(img_clahe)
    haralick_feat = extract_haralick(img_clahe)
    fused = np.concatenate([deep_feat, lbp_feat, haralick_feat]).reshape(1, -1)

    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        fused = scaler.transform(fused)

    proba = clf.predict_proba(fused)[0]
    pred = int(np.argmax(proba))

    overlay, _ = grad_cam(img_bgr)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUTS_DIR, "gradcam_overlay.png")
    cv2.imwrite(out_path, overlay)
    return pred, proba, out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()
    p, pr, hp = infer_image(args.image)
    print("Prediction:", p)
    print("Probabilities:", pr)
    print("Grad-CAM saved to:", hp)
