import os
import cv2
import numpy as np
import pandas as pd
from . import config

def _sanitize_id_code(id_code):
    """
    Handle Excel-corrupted scientific notation (e.g., '7.10E+10') and float ids.

    Args:
        id_code: Raw ID code that may be corrupted

    Returns:
        str: Sanitized ID code as string
    """
    s = str(id_code).strip()

    # Validate that id_code contains only expected characters
    if not s or len(s) > 20:
        raise ValueError(f"Invalid ID code format: {id_code}")

    # Detect scientific notation or float-like ids (common Excel corruption)
    if "e+" in s.lower() or "e-" in s.lower():
        try:
            # Try to recover integer representation (lossy but better than FileNotFound)
            # e.g., '7.10E+10' -> 71000000000
            s = str(int(float(s)))
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Could not parse corrupted ID code: {id_code}") from e

    # Remove trailing .0 if pandas inferred float
    if s.endswith(".0"):
        s = s[:-2]

    # Final validation: must be digits only
    if not s.isdigit():
        raise ValueError(f"Invalid ID code (non-numeric after sanitization): {id_code}")

    return s

def load_labels():
    # Force id_code to string to prevent pandas float inference on corrupted rows
    df = pd.read_csv(config.LABELS_FILE, dtype={"id_code": str})
    df["id_code"] = df["id_code"].apply(_sanitize_id_code)
    # Log corrupted cases
    corrupted = df[df["id_code"].str.contains(r"[eE]\+", na=False)]
    if not corrupted.empty:
        print(f"Warning: {len(corrupted)} id_code entries still look corrupted and will be skipped if images missing: {corrupted['id_code'].head().tolist()}")
    return df

def image_path(id_code):
    sid = _sanitize_id_code(id_code)
    return os.path.join(config.DATA_DIR, f"{sid}.png")

def advanced_preprocess_image(image_data, target_size=None, from_numpy=False):
    """
    Read and preprocess image: resize, BGR->GRAY, apply CLAHE.

    Args:
        image_data (str or ndarray): Path to image file or numpy array
        target_size (tuple): Target size (width, height). Defaults to config.TARGET_SIZE
        from_numpy (bool): If True, treat image_data as numpy array instead of path

    Returns:
        tuple: (img_bgr_resized, gray_clahe) processed images

    Raises:
        FileNotFoundError: If image path doesn't exist or image cannot be read
    """
    if target_size is None:
        target_size = config.TARGET_SIZE
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
