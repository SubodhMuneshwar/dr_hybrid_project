import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

from . import config
from .data import load_labels, image_path, advanced_preprocess_image
from .features import get_deep_feature_model, extract_deep_features, extract_lbp, extract_haralick
from .models import split_and_scale, get_base_pipelines, tune_model, build_stacking, save_model
from .evaluate import save_report

def _build_or_load_features(labels_df):
    os.makedirs(os.path.dirname(config.FEATURES_CACHE), exist_ok=True)
    if os.path.exists(config.FEATURES_CACHE):
        cache = np.load(config.FEATURES_CACHE, allow_pickle=True)
        print(f"✅ Loaded features from cache: {config.FEATURES_CACHE}")
        return cache["X"], cache["y"]

    deep_model, preprocess_fn = get_deep_feature_model(config.FEATURE_EXTRACTOR_MODEL)
    X, y = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting features"):
        p = image_path(row["id_code"])
        if not os.path.exists(p):
            continue
        img_bgr, img_clahe = advanced_preprocess_image(p, target_size=(224, 224))
        deep_feat = extract_deep_features(img_bgr, deep_model, preprocess_fn)
        lbp_feat = extract_lbp(img_clahe)
        haralick_feat = extract_haralick(img_clahe)
        X.append(np.concatenate([deep_feat, lbp_feat, haralick_feat]))
        y.append(int(row["diagnosis"]))

    X = np.array(X); y = np.array(y)
    np.savez_compressed(config.FEATURES_CACHE, X=X, y=y)
    print(f"✅ Saved features to: {config.FEATURES_CACHE}")
    return X, y

def train():
    labels = load_labels()
    X, y = _build_or_load_features(labels)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    (rf, rf_params), (svm, svm_params), (knn, knn_params) = get_base_pipelines()
    rf_best = tune_model(rf, rf_params, X_train, y_train)
    svm_best = tune_model(svm, svm_params, X_train, y_train)
    knn_best = tune_model(knn, knn_params, X_train, y_train)

    stack = build_stacking(rf_best, svm_best, knn_best)
    stack.fit(X_train, y_train)

    model_path = os.path.join(config.MODELS_DIR, "stacking_calibrated.pkl")
    save_model(stack, model_path)
    print(f"✅ Model saved to: {model_path}")

    y_pred = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)
    txt, cm, f1 = save_report(y_test, y_pred, y_proba, config.OUTPUTS_DIR, config.CLASS_NAMES, prefix="stacking")
    print(f"✅ Evaluation saved:\n- {txt}\n- {cm}\n- {f1}")

def evaluate():
    cache = np.load(config.FEATURES_CACHE, allow_pickle=True)
    X, y = cache["X"], cache["y"]
    from .models import load_model, split_and_scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    model = load_model(os.path.join(config.MODELS_DIR, "stacking_calibrated.pkl"))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    txt, cm, f1 = save_report(y_test, y_pred, y_proba, config.OUTPUTS_DIR, config.CLASS_NAMES, prefix="stacking_reval")
    print(f"✅ Re-evaluation saved:\n- {txt}\n- {cm}\n- {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.evaluate:
        evaluate()
    else:
        print("Usage: python -m src.pipeline --train [--evaluate]")
