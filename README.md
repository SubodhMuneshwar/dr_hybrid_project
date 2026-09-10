# 🏥 RETINASCAN: Diabetic Retinopathy Hybrid Detection

A professional, clinical-grade medical diagnostic system for detecting Diabetic Retinopathy using retinal fundus images and a Hybrid Feature Fusion ensemble.

![Medical UI](https://raw.githubusercontent.com/your-username/dr-hybrid-project/main/app/static/preview.png) *(Add a screenshot here later)*

---

## 🚀 System Overview

This project implements a **two-stage diagnostic pipeline** combining Deep Learning Shape Analysis with Mathematical Texture Engineering.

### 🧠 Stage 1: The Hybrid Model
The engine consists of an ensemble of models trained on thousands of Kaggle fundus images:
- **Feature Fusion:** Combines **DenseNet121** deep visual features with **LBP & Haralick** texture descriptors.
- **Ensemble Logic:** A **Voting Classifier** (SVM + Random Forest + KNN) makes the final clinical determination.
- **SMOTE Balancing:** Applied to handle class imbalance across the 5 stages of DR.

### 💻 Stage 2: Clinical Interface
The web interface is designed for real-world clinicians:
- **Diagnostic View:** Interactive viewport with **Scanning Brackets** and pulsing AI activation.
- **Explainability:** Integrated **Grad-CAM** heatmaps that highlight disease indicators (Microaneurysms, Hemorrhages).
- **Analytics Dashboard:** Full visualization of model F1-scores, accuracy bar charts, and error matrices.

---

## 🛠️ Setup & Installation

Follow these steps to set up the project on your local machine:

### 1. Prerequisites
- **Python 3.9+**
- Git installed on your system.

### 2. Clone and Environment Setup
```bash
# Clone the repository
git clone https://github.com/Nihar0001/dr_hybrid_project.git
cd dr_hybrid_project

# Create a virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Running the Application
```bash
# Set Flask environment
# Windows (PowerShell):
$env:FLASK_APP="app/app.py"
# Windows (CMD):
set FLASK_APP=app/app.py
# Mac/Linux:
export FLASK_APP=app/app.py

# Launch the server
flask run --port 5001
```
Access the system at: `http://127.0.0.1:5001`

---

## 📂 Project Structure

- `app/`: Flask web application, clinical templates, and styles.
- `src/`: Core ML logic, feature extraction, and Grad-CAM implementation.
- `models/`: Pre-trained Voting Classifier and Scalers.
- `data/`: Dataset CSVs (train.csv) and reference images.
- `outputs/`: Generated diagnostic reports and overlay snapshots.

---

### 🚨 Important: Manual Model Setup
> [!CAUTION]
> **The Model Files are MISSING from this repository.** 
> Because the `votingclassifier_model.pkl` is **2.6GB**, it exceeds GitHub's file size limit (100MB).

**To make the project run, you must:**
1. Download the `models/` folder manually from this [Google Drive Link](https://drive.google.com/drive/folders/1ObEF3nNfyCsRqXyNYNNnEfshAwr2dgi6?usp=sharing).
2. Place the `.pkl` files inside the `models/` directory of your local clone.

```bash
# Required files:
# models/votingclassifier_model.pkl (2.6GB)
# models/scaler.pkl
```

## 🤝 For Team Members
To edit the design or logic:
- **CSS Styles:** Modify `app/static/styles.css` to update the Clinical Design System.
- **HTML Layouts:** Modify `app/templates/scanner.html` or `dashboard.html`.
- **Inference Logic:** See `src/infer.py` for how the model processes new images.

---

## ⚖️ License
This project is for educational and research purposes.