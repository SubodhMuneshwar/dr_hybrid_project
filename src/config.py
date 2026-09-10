import os

# ---------- Paths ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train_images")
LABELS_FILE = os.path.join(PROJECT_ROOT, "data", "train.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
FEATURES_CACHE = os.path.join(OUTPUTS_DIR, "features_cache.npz")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------- Feature extractor ----------
# Options: 'mobilenetv2', 'densenet121', 'inceptionresnetv2', 'vgg16'
FEATURE_EXTRACTOR_MODEL = "vgg16"

# ---------- Classes ----------
# Diabetic Retinopathy Classification Levels
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

CLASS_DESCRIPTIONS = {
    0: "No Diabetic Retinopathy detected. The retina appears healthy.",
    1: "Mild DR: Small dot and blot hemorrhages may be present.",
    2: "Moderate DR: More extensive hemorrhages and microaneurysms are visible.",
    3: "Severe DR: Numerous hemorrhages and venous beading indicating advanced damage.",
    4: "Proliferative DR: New blood vessel growth (neovascularization) is present - requires immediate attention."
}
