# RetinaScan — Diabetic Retinopathy Detection System

> A Flask-based retinal image analysis and diagnostic interface integrating a hybrid machine-learning inference pipeline.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#license)

## Overview

RetinaScan is an educational/research project for analyzing retinal fundus images for signs of diabetic retinopathy (DR).

The system integrates a hybrid feature-fusion inference pipeline into a Flask web application. A user uploads a retinal image, the backend extracts and fuses deep and texture features, runs a trained classifier, and returns a predicted DR stage with a probability distribution. Where supported, the application also generates a Grad-CAM-based activation overlay to help explain where the feature extractor responded most strongly.

This is not a clinical diagnostic product — predictions are for research and demonstration purposes only.

## My Role

I owned the frontend development and application/system integration for this project.

My contributions focused on the application layer that connects the user to the inference pipeline:

- Designed and implemented the web application's frontend and interface (landing page, diagnostic view, dashboard)
- Built the retinal image upload workflow with input validation and user feedback
- Integrated the Flask application with the inference pipeline (`src/infer.py`)
- Connected prediction outputs to the user interface
- Implemented prediction and result presentation, including confidence/probability display where supported by the inference output
- Integrated Grad-CAM output into the web workflow and surfaced it in the diagnostic view
- Worked on the application flow connecting image input, backend inference, and displayed results
- Contributed to the overall user experience, error handling, and presentation layer

I did not independently train the ML models. Model architecture design, training, and evaluation were handled collaboratively by the project team (see below).

## Team Contribution

This project was developed collaboratively. The ML/model-development work included the training and evaluation of the hybrid classification pipeline, while I owned the frontend and application integration layer.

The hybrid pipeline combines deep features with handcrafted texture descriptors and a classical ML classifier. The repository retains this design as the inference path consumed by the Flask app. Specific training and experimentation were carried out by the team; this README documents the architecture as it exists in the current codebase.

## System Architecture

```
Retinal Image
    ↓
Preprocessing                (resize to TARGET_SIZE, CLAHE — src/data.py)
    ↓
Feature Extraction
    ├── DenseNet121 deep features  }  configurable via FEATURE_EXTRACTOR_MODEL
    ├── LBP texture features       }  (current default: VGG16 GAP 512-d; also supports
    └── Haralick texture features }   DenseNet121 1024-d, MobileNetV2, InceptionResNetV2)
    ↓
Feature Fusion / Scaling     (concatenation → StandardScaler from models/scaler.pkl)
    ↓
ML Classifier                (stacking_calibrated.pkl preferred; votingclassifier_model.pkl fallback)
    ↓
Prediction + Probability     (predict_proba → argmax)
    ↓
Flask Application            (app/app.py — upload, validation, inference, rendering)
    ↓
Diagnostic Result / Visualization   (predicted stage, confidence, class descriptions)
    ↓
Grad-CAM Output              (activation feature map → outputs/gradcam_overlay.png)
```

> Note: `src/features.py:get_deep_feature_model()` supports `vgg16`, `densenet121`, `mobilenetv2`, and `inceptionresnetv2`. The current default in `src/config.py` is `FEATURE_EXTRACTOR_MODEL = "vgg16"` with `TARGET_SIZE = (224, 224)` and Global Average Pooling (512-d). The diagram above reflects the hybrid design as described in the project history; the extractor is swappable without changing the downstream pipeline.

## ML Pipeline

Verified against the current repository (`src/infer.py`, `src/features.py`, `src/data.py`, `src/models.py`, `src/pipeline.py`, `src/explain.py`, `src/config.py`):

1. **Preprocessing** — `advanced_preprocess_image()` resizes to `TARGET_SIZE` (224x224) and applies CLAHE on the grayscale channel. Returns both the resized BGR image (for deep features) and the CLAHE-enhanced grayscale image (for texture features).

2. **Deep feature extraction** — Loads a pretrained ImageNet backbone via `get_deep_feature_model()` with `include_top=False` + `GlobalAveragePooling2D`. Preprocessing function is matched to the backbone (e.g., `vgg_preprocess`, `densenet_preprocess`).

3. **LBP texture descriptors** — `extract_lbp()` computes uniform LBP (radius 3, 24 points) and returns a 26-bin normalized histogram.

4. **Haralick texture descriptors** — `extract_haralick()` computes GLCM (distances=[1], 4 angles) and concatenates 6 properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM) → 24-d.

5. **Feature fusion** — Deep + LBP + Haralick concatenated into a single vector (e.g., 512 + 26 + 24 = 562-d for VGG16 GAP).

6. **Feature scaling** — `models/scaler.pkl` (`StandardScaler`) is loaded and applied if present (`infer.py:35-38`).

7. **Classifier inference** — `_load_classifier()` checks in order:
   - `models/stacking_calibrated.pkl` (preferred — `StackingClassifier` with RF + SVM + KNN base estimators and Logistic Regression meta-learner, `src/models.py:build_stacking`)
   - `models/votingclassifier_model.pkl` (fallback for backward compatibility)

   Both are trained with SMOTE inside `ImbPipeline` to address class imbalance (`src/models.py:get_base_pipelines`).

8. **Probability output** — `clf.predict_proba()` returns a 5-class distribution over `["No DR", "Mild", "Moderate", "Severe", "Proliferative"]` (`config.CLASS_NAMES`).

9. **Grad-CAM / activation map** — `explain.grad_cam()` builds a VGG16 `block5_conv3` activation map (mean across 512 filters), normalizes, color-maps with `COLORMAP_JET`, and overlays onto the input image. Saved to `outputs/gradcam_overlay.png` and served to the UI. When TensorFlow is unavailable, a passthrough fallback is returned so the Flask app can still load.

Training and evaluation are driven by `python -m src.pipeline --train` / `--evaluate`, which handle feature caching (`outputs/features_cache.npz`), train/test splitting, hyperparameter search, stacking fit, and report generation (`src/evaluate.py` → confusion matrix and F1-score plots).

## Web Application

The application layer is implemented in `app/app.py` — this is the code I owned and integrated.

- **Flask-based web application** with three routes: `/` (landing), `/scanner` (diagnostic view), `/dashboard` (analytics)
- **Retinal image upload** — `POST /scanner` with `multipart/form-data`, file input `name="file"`
- **File validation** — extension allowlist (`png`, `jpg`, `jpeg`), `secure_filename`, 10 MB `MAX_CONTENT_LENGTH`, empty-file checks with `flash()` feedback
- **Image processing / inference workflow** — saves upload to `uploads/`, calls `src.infer.infer_image()`, which runs the full pipeline described above
- **Prediction output** — predicted class index, mapped to `CLASS_NAMES` and `CLASS_DESCRIPTIONS`
- **Confidence / probability display** — `confidence = proba[pred] * 100` formatted to one decimal place, plus the full 5-class `proba` array for the UI
- **Diagnostic result presentation** — rendered in `scanner.html` with original image and overlay side-by-side
- **Grad-CAM visualization** — overlay written to `outputs/gradcam_overlay.png`, served via `/outputs/<path:filename>` and referenced as `overlay_url` in the template
- **Dashboard / analysis views** — `dashboard.html` displays confusion-matrix and F1-score artifacts from `outputs/` with graceful fallback when artifacts are absent (shows "Awaiting Model Metadata...")
- **Error handling and user feedback** — `try/except` around `infer_image()` with flashed inference errors and redirects

Templates use Tailwind CSS (via CDN), Inter font, and glass-morphism styling. Static assets live in `app/static/`.

## Key Features

### Application

- Retinal image upload with drag-and-drop and click-to-browse
- Input validation (file type, size, presence) with flashed error messages
- End-to-end prediction workflow (upload → preprocessing → feature extraction → inference → rendering)
- Probability/confidence presentation per predicted stage
- Diagnostic result visualization with class name and clinical description
- Grad-CAM activation overlay generation and display
- Analytics dashboard that surfaces confusion-matrix and F1-score plots when available
- Uploaded image and overlay served side-by-side in the diagnostic view

### Engineering

- Flask application structure with configurable `PORT` and `FLASK_DEBUG` env vars
- Modular inference pipeline (`src/data.py`, `src/features.py`, `src/models.py`, `src/infer.py`, `src/explain.py`)
- Swappable deep feature extractor (VGG16 / DenseNet121 / MobileNetV2 / InceptionResNetV2)
- Feature extraction components (deep + LBP + Haralick) with unified `TARGET_SIZE`
- Feature fusion and `StandardScaler` abstraction
- Classifier abstraction with ordered fallback (`stacking_calibrated.pkl` → `votingclassifier_model.pkl`)
- Generated output artifacts (`outputs/gradcam_overlay.png`, `stacking_confusion_matrix.png`, `stacking_f1_scores.png`, `stacking_report.txt`, `features_cache.npz`)
- Data-corruption handling for Excel-mangled `id_code` values (`src/data.py:_sanitize_id_code`)

## Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python |
| Web Framework | Flask, Werkzeug |
| Computer Vision | OpenCV (CLAHE, image I/O) |
| Deep Learning | TensorFlow / Keras — VGG16 (default), DenseNet121, MobileNetV2, InceptionResNetV2 (all GAP) |
| Feature Engineering | LBP (scikit-image `local_binary_pattern`), Haralick / GLCM (scikit-image `graycomatrix` / `graycoprops`) |
| ML | scikit-learn (StackingClassifier, RandomForest, SVC, KNN, LogisticRegression, GridSearchCV), imbalanced-learn (SMOTE, ImbPipeline), joblib |
| Data | NumPy, Pandas, tqdm |
| Explainability | Grad-CAM activation map (VGG16 `block5_conv3`) |
| Visualization | Matplotlib, Seaborn (evaluation reports) |
| Version Control | Git, GitHub |

Dependencies are pinned in `requirements.txt`. No additional libraries are required beyond what is listed there.

## Project Scale

> **10,000+ retinal images**

The hybrid pipeline was designed to operate across 10,000+ retinal images. The current `data/train.csv` included in the repository contains the labeled subset used for local development and evaluation (with `data/train_images/` for the corresponding image files when present locally).

## Screenshots

Screenshots below are from the actual application templates and generated artifacts. Image files are served from `app/static/` and `outputs/`.

### Home / Landing Page

`app/templates/index.html` — hero section with scanner CTA and analytics entry point.

> To add a screenshot: save it as `app/static/screenshot_home.png` and reference it as `![Home](app/static/screenshot_home.png)`.

### Diagnostic View

`app/templates/scanner.html` — upload area, original vs. Grad-CAM overlay, prediction and confidence.

> To add a screenshot: save it as `app/static/screenshot_scanner.png` and reference it as `![Scanner](app/static/screenshot_scanner.png)`.

### Grad-CAM Overlay (generated artifact)

This file is produced by every inference run and also ships as a static preview:

![Grad-CAM overlay](app/static/gradcam_overlay.png)

Source: `app/static/gradcam_overlay.png` (preview) and `outputs/gradcam_overlay.png` (per-inference output, served via `/outputs/gradcam_overlay.png`).

### Analytics Dashboard

`app/templates/dashboard.html` — confusion matrix and F1-score visualizations when `outputs/stacking_confusion_matrix.png` and `outputs/stacking_f1_scores.png` are present; otherwise displays "Awaiting Model Metadata..." placeholders.

> To add dashboard screenshots: run `python -m src.pipeline --train` or `--evaluate` to generate `outputs/stacking_confusion_matrix.png` and `outputs/stacking_f1_scores.png`, then reference them via the `/outputs/<filename>` route.

## Project Structure

```
dr_hybrid_project/
├── app/
│   ├── app.py                 # Flask application — upload, inference, result rendering (my integration layer)
│   ├── templates/
│   │   ├── index.html         # Landing page
│   │   ├── scanner.html       # Diagnostic view — upload + results + Grad-CAM
│   │   └── dashboard.html     # Analytics dashboard — confusion matrix / F1 views
│   └── static/
│       ├── styles.css
│       ├── gradcam_overlay.png    # Preview / fallback overlay image
│       └── uploads/               # Served originals (gitignored except .gitkeep)
├── src/
│   ├── config.py              # Paths, TARGET_SIZE, FEATURE_EXTRACTOR_MODEL, class names
│   ├── data.py                # Label loading, id_code sanitization, preprocessing (CLAHE)
│   ├── features.py            # Deep feature extractor + LBP + Haralick
│   ├── models.py              # SMOTE pipelines, GridSearchCV, StackingClassifier
│   ├── infer.py               # Inference entry point — fusion, scaling, classifier, Grad-CAM
│   ├── explain.py             # Grad-CAM activation map (VGG16 block5_conv3)
│   ├── evaluate.py            # Classification report, confusion matrix, F1 plots
│   └── pipeline.py            # Training / evaluation orchestration (python -m src.pipeline)
├── data/
│   ├── train.csv              # Labels (id_code, diagnosis) — images in train_images/ when present
│   └── train_images/          # Retinal images (gitignored except .gitkeep)
├── models/
│   ├── scaler.pkl             # StandardScaler (present; 770 KB)
│   └── stacking_calibrated.pkl / votingclassifier_model.pkl  # Trained classifiers (see Model Files)
├── outputs/                   # Generated artifacts — gradcam_overlay.png, *_confusion_matrix.png, *_f1_scores.png, features_cache.npz (gitignored)
├── uploads/                   # Temporary Flask uploads (gitignored)
├── archive/                   # Historical evaluation notes and helper scripts (not part of runtime)
├── requirements.txt
├── generate_scaler.py
├── setup_domain.py
└── README.md
```

> The repository does not contain top-level `models/` or `outputs/` artifacts on GitHub beyond `.gitkeep` — large `.pkl` files are gitignored and must be placed locally (see Model Files).

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/SubodhMuneshwar/dr_hybrid_project.git
cd dr_hybrid_project

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> Note: TensorFlow (`tensorflow>=2.12`) requires Python 3.10–3.11 on Windows. If you are on Python 3.12+, the Flask UI will still load (feature extraction stubs gracefully), but inference will require a compatible Python version.

## Model Files

Trained model artifacts are too large to store directly on GitHub and are gitignored (`models/*.pkl` in `.gitignore`).

### Required files

| File | Purpose | Status |
|---|---|---|
| `models/scaler.pkl` | `StandardScaler` fitted on fused features | Included locally (770 KB); regenerate via `python -m src.pipeline --train` if missing |
| `models/stacking_calibrated.pkl` | **Preferred** classifier — `StackingClassifier` (RF + SVM + KNN → Logistic Regression) | Required for current inference path; download or train locally |
| `models/votingclassifier_model.pkl` | **Fallback** classifier — legacy voting ensemble | Retained for backward compatibility; used only if `stacking_calibrated.pkl` is absent |

`src/infer.py:_load_classifier()` checks for `stacking_calibrated.pkl` first, then falls back to `votingclassifier_model.pkl`. If neither is present, inference raises `FileNotFoundError` with instructions.

### Where to place them

Place the `.pkl` files directly in the `models/` directory at the repository root:

```
dr_hybrid_project/
└── models/
    ├── scaler.pkl
    ├── stacking_calibrated.pkl   # preferred
    └── votingclassifier_model.pkl  # fallback
```

### Where to download

The legacy voting classifier and scaler were previously distributed via Google Drive (link from the prior README):

- https://drive.google.com/drive/folders/1ObEF3nNfyCsRqXyNYNNnEfshAwr2dgi6?usp=sharing

If a stacking-calibrated artifact is published, it should be placed at `models/stacking_calibrated.pkl` in the same directory. Alternatively, train it locally:

```bash
python -m src.pipeline --train
```

This generates `models/stacking_calibrated.pkl` and `models/scaler.pkl`, plus evaluation artifacts in `outputs/`.

> Do not commit `.pkl` files to Git — they are intentionally gitignored due to size (the legacy voting model is ~2.5 GB).

## Running the Application

```bash
# From the repository root, with the virtual environment activated
# and model files in place (see Model Files above):

# Option A — Flask CLI (recommended)
# Windows (PowerShell)
$env:FLASK_APP="app/app.py"
flask run --port 5001

# Windows (CMD)
set FLASK_APP=app/app.py
flask run --port 5001

# macOS / Linux
export FLASK_APP=app/app.py
flask run --port 5001

# Option B — Direct Python
python app/app.py
# respects PORT and FLASK_DEBUG env vars (defaults: 5001, false)

# Then open:
# http://127.0.0.1:5001          → Landing page
# http://127.0.0.1:5001/scanner  → Diagnostic view (upload + results)
# http://127.0.0.1:5001/dashboard → Analytics dashboard
```

Additional commands:

```bash
# Run inference on a single image (without the web UI)
python -m src.infer --image path/to/image.png

# Re-train the stacking pipeline (requires data/train_images/ and data/train.csv)
python -m src.pipeline --train

# Re-evaluate a cached feature set
python -m src.pipeline --evaluate
```

Performance metrics depend on the trained model artifact and evaluation configuration included with the project.

## Limitations

- Trained model artifacts (`*.pkl`) are not stored directly on GitHub because of file-size constraints and must be downloaded or trained locally before inference will run.
- Local inference requires the trained model files in `models/` and a Python environment with the dependencies in `requirements.txt` (TensorFlow requires Python 3.10–3.11 for full feature extraction).
- This is an educational/research project — predictions should not be treated as medical diagnosis and the system is not intended for clinical use or as a production medical device.
- The Grad-CAM output in `src/explain.py` is an activation feature map (mean of VGG16 `block5_conv3` filters), not a gradient-based explanation of the sklearn classifier's decision boundary.
- Dashboard visualizations (`outputs/*.png`) only appear after a training or evaluation run has generated them.

## Collaboration & Attribution

This project was developed collaboratively. I owned the frontend implementation and application/system integration, while the ML pipeline development, model training, and evaluation were handled collaboratively by the project team.

The repository is maintained and documented under my GitHub account at https://github.com/SubodhMuneshwar/dr_hybrid_project.

## License

This project is for educational and research purposes.
