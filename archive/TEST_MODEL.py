"""
Comprehensive Model Testing & Evaluation Script
Tests all capabilities of your trained Stacking Classifier
"""

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, 
    cohen_kappa_score, hamming_loss
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))

from src import config
from src.models import load_model, split_and_scale
from src.data import load_labels, image_path, advanced_preprocess_image
from src.features import get_deep_feature_model, extract_deep_features, extract_lbp, extract_haralick

def load_test_features():
    """Load or regenerate test data."""
    labels = load_labels()
    X, y = [], []
    
    print("Loading test images...")
    deep_model, preprocess_fn = get_deep_feature_model(config.FEATURE_EXTRACTOR_MODEL)
    
    for idx, (_, row) in enumerate(labels.iterrows()):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(labels)}...")
        
        p = image_path(row["id_code"])
        if not os.path.exists(p):
            continue
        
        try:
            img_bgr, img_clahe = advanced_preprocess_image(p, target_size=(224, 224))
            deep_feat = extract_deep_features(img_bgr, deep_model, preprocess_fn)
            lbp_feat = extract_lbp(img_clahe)
            haralick_feat = extract_haralick(img_clahe)
            X.append(np.concatenate([deep_feat, lbp_feat, haralick_feat]))
            y.append(int(row["diagnosis"]))
        except:
            continue
    
    return np.array(X), np.array(y)

def test_model_comprehensive():
    """Test all aspects of model performance."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*70)
    
    # Load model
    model_path = os.path.join(config.MODELS_DIR, "stacking_calibrated.pkl")
    print(f"\nLoading model from: {model_path}")
    clf = joblib.load(model_path)
    
    # Load and prepare test data
    X, y = load_test_features()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(y)}")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Classes: {config.CLASS_NAMES}")
    print(f"  Class distribution (test): {np.bincount(y_test)}")
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # ==================== ACCURACY METRICS ====================
    print("\n" + "-"*70)
    print("1. OVERALL ACCURACY METRICS")
    print("-"*70)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"Accuracy (overall): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    
    # ==================== PER-CLASS METRICS ====================
    print("\n" + "-"*70)
    print("2. PER-CLASS PERFORMANCE (Precision, Recall, F1)")
    print("-"*70)
    
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print(report)
    
    # ==================== CONFIDENCE ANALYSIS ====================
    print("\n" + "-"*70)
    print("3. CONFIDENCE ANALYSIS")
    print("-"*70)
    
    max_proba = np.max(y_proba, axis=1)
    
    print(f"Average Confidence: {np.mean(max_proba):.4f}")
    print(f"Min Confidence: {np.min(max_proba):.4f}")
    print(f"Max Confidence: {np.max(max_proba):.4f}")
    print(f"Confidence Std Dev: {np.std(max_proba):.4f}")
    
    # Confidence bins
    confidence_bins = [0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"\nConfidence Distribution:")
    for i in range(len(confidence_bins)-1):
        count = np.sum((max_proba >= confidence_bins[i]) & (max_proba < confidence_bins[i+1]))
        pct = (count / len(max_proba)) * 100
        print(f"  {confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}: {count:4d} ({pct:5.1f}%)")
    
    # ==================== CONFIDENCE VS ACCURACY ====================
    print("\n" + "-"*70)
    print("4. CONFIDENCE CALIBRATION")
    print("-"*70)
    
    correct = (y_pred == y_test)
    correct_confidence = max_proba[correct]
    incorrect_confidence = max_proba[~correct]
    
    print(f"Average confidence when CORRECT: {np.mean(correct_confidence):.4f}")
    print(f"Average confidence when WRONG: {np.mean(incorrect_confidence):.4f}")
    print(f"Difference (should be high): {np.mean(correct_confidence) - np.mean(incorrect_confidence):.4f}")
    
    # ==================== CONFUSION MATRIX ====================
    print("\n" + "-"*70)
    print("5. CONFUSION MATRIX (normalized)")
    print("-"*70)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    
    print("(Rows = True Class, Columns = Predicted Class)")
    print(cm_norm)
    
    # Save visualizations
    visualize_results(y_test, y_pred, y_proba, cm_norm)
    
    # ==================== CLASS-WISE CAPABILITY ====================
    print("\n" + "-"*70)
    print("6. WHICH CLASSES CAN THE MODEL PREDICT WELL?")
    print("-"*70)
    
    for i, class_name in enumerate(config.CLASS_NAMES):
        class_mask = y_test == i
        if class_mask.sum() == 0:
            continue
        class_indices = np.where(class_mask)[0]
        class_accuracy = np.mean(y_pred[class_mask] == i)
        avg_confidence = np.mean(y_proba[class_mask, i])
        
        status = "✅ GOOD" if class_accuracy > 0.75 else "⚠️  FAIR" if class_accuracy > 0.50 else "❌ POOR"
        print(f"{class_name:20s}: {class_accuracy*100:5.1f}% accuracy, {avg_confidence*100:5.1f}% avg confidence {status}")
    
    # ==================== WORST PREDICTIONS ====================
    print("\n" + "-"*70)
    print("7. HARDEST SAMPLES (Lowest Confidence Errors)")
    print("-"*70)
    
    errors = y_pred != y_test
    if errors.sum() > 0:
        error_confidence = max_proba[errors]
        error_indices = np.where(errors)[0]
        sorted_indices = error_indices[np.argsort(error_confidence)][:10]  # 10 hardest
        
        print(f"Total errors: {errors.sum()} out of {len(y_test)}")
        print("\nHardest 10 predictions:")
        for idx, sample_idx in enumerate(sorted_indices):
            print(f"  {idx+1}. True: {config.CLASS_NAMES[y_test[sample_idx]]}, "
                  f"Pred: {config.CLASS_NAMES[y_pred[sample_idx]]}, "
                  f"Conf: {max_proba[sample_idx]:.2f}")
    else:
        print("No errors! Model is perfect on test set.")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE - Results saved to: outputs/model_test_results/")
    print("="*70 + "\n")

def visualize_results(y_test, y_pred, y_proba, cm_norm):
    """Create and save visualizations."""
    
    output_dir = os.path.join(config.OUTPUTS_DIR, "model_test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                cbar_kws={'label': 'Normalized Count'})
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 2. Confidence Distribution
    max_proba = np.max(y_proba, axis=1)
    plt.figure(figsize=(10, 5))
    plt.hist(max_proba, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(max_proba), color='r', linestyle='--', label=f'Mean: {np.mean(max_proba):.3f}')
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Confidence Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
    plt.close()
    
    # 3. Per-class accuracy
    class_accuracies = []
    for i in range(len(config.CLASS_NAMES)):
        mask = y_test == i
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == i)
        else:
            acc = 0
        class_accuracies.append(acc)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(config.CLASS_NAMES, class_accuracies, color='steelblue', edgecolor='black')
    plt.axhline(0.75, color='g', linestyle='--', alpha=0.7, label='Good (75%)')
    plt.axhline(0.50, color='orange', linestyle='--', alpha=0.7, label='Fair (50%)')
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=150)
    plt.close()
    
    print(f"\n✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    test_model_comprehensive()
