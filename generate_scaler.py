import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src import config

def main():
    print("Loading original unscaled training data...")
    # Load original training data
    train_data_path = "D:/all mini projects(codes)/Enhancing-diabetic-retinopathy-detection/data/X_train.npy"
    if not os.path.exists(train_data_path):
        print(f"Error: {train_data_path} not found.")
        return
        
    X_train = np.load(train_data_path)
    print(f"Loaded X_train with shape: {X_train.shape}")
    
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    main()
