import os
import cv2
import numpy as np
import pandas as pd
from . import config

def load_labels():
    df = pd.read_csv(config.LABELS_FILE)
    # Kaggle-style: id_code (png file name without extension), diagnosis (0-4)
    return df

def image_path(id_code):
    return os.path.join(config.DATA_DIR, f"{id_code}.png")

def advanced_preprocess_image(image_data, target_size=(224, 224), from_numpy=False):
    """
    Read image path or array, resize, BGR->GRAY -> CLAHE.
    Returns: (img_bgr_resized, gray_clahe)
    """
    if from_numpy:
        img_bgr = image_data
    else:
        img_bgr = cv2.imread(image_data)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_data}")

    img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    return img_bgr, img_clahe
