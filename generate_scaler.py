import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src import config

def main():
    print("Loading training data...")
    # Try cached features first, then legacy path for backward compat
    candidates = [
        config.FEATURES_CACHE,
        os.path.join(config.DATA_DIR, "X_train.npy"),
        os.path.join(config.OUTPUTS_DIR, "features_cache.npz"),
    ]
    # Also allow X_train.npy in project root data/
    X_train = None
    train_data_path = None
    for p in candidates:
        if p and os.path.exists(p):
            train_data_path = p
            break
    if train_data_path is None:
        print(f"Error: No training data found. Checked: {candidates}")
        print("Run: python -m src.pipeline --train  to generate features_cache.npz first")
        return

    if train_data_path.endswith(".npz"):
        data = np.load(train_data_path, allow_pickle=True)
        # features_cache.npz contains X; X_train.npy is raw array
        if "X" in data:
            X_train = data["X"]
        else:
            X_train = data[data.files[0]]
    else:
        X_train = np.load(train_data_path)
    print(f"Loaded X_train from {train_data_path} with shape: {X_train.shape}")
    
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    main()
