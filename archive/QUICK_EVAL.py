"""
QUICK MODEL EVALUATION
Fast test without reprocessing all images
"""

import os
import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from src import config
from src.models import split_and_scale

# Load cached features if available
cache_path = config.FEATURES_CACHE
if os.path.exists(cache_path):
    print(f"Loading cached features from: {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    X, y = cache['X'], cache['y']
    
    # Split data
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    # Load and test model
    model_path = os.path.join(config.MODELS_DIR, "stacking_calibrated.pkl")
    print(f"\nTesting model: {model_path}\n")
    
    clf = joblib.load(model_path)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Report
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"QUICK EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"F1-Score (macro):    {f1_macro:.4f}")
    print(f"\nPER-CLASS PERFORMANCE:")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES))
    
    # Confidence analysis
    max_proba = np.max(y_proba, axis=1)
    correct = y_pred == y_test
    
    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Average confidence:      {np.mean(max_proba):.4f}")
    print(f"  Avg confidence (correct): {np.mean(max_proba[correct]):.4f}")
    print(f"  Avg confidence (wrong):   {np.mean(max_proba[~correct]):.4f}")
    
    print(f"\nPER-CLASS ACCURACY:")
    for i, name in enumerate(config.CLASS_NAMES):
        mask = y_test == i
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == i)
            print(f"  {name:20s}: {acc*100:5.1f}%")
    
    print(f"\n{'='*60}")
    
else:
    print(f"❌ Cache not found: {cache_path}")
    print("Run TEST_MODEL.py first to extract features")
