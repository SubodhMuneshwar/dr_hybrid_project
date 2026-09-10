import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from . import config

def save_report(y_true, y_pred, y_proba, out_dir, class_names, prefix="stacking"):
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # write text report
    txt_path = os.path.join(out_dir, f"{prefix}_report.txt")
    with open(txt_path, "w") as f:
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    # confusion matrix plot (normalized)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()

    # F1 scores bar chart
    # parse from classification_report
    lines = [l for l in report.splitlines() if l.strip() and l.strip()[0].isdigit()]
    f1_scores = []
    labels = []
    for line in lines[:len(class_names)]:
        parts = line.split()
        labels.append(parts[0])
        f1_scores.append(float(parts[-2]))  # support is last, f1 is second last

    plt.figure(figsize=(7,4))
    sns.barplot(x=labels, y=f1_scores)
    plt.ylim(0, 1.0)
    plt.title("F1-Score by Class")
    plt.xlabel("Class")
    plt.ylabel("F1-score")
    f1_path = os.path.join(out_dir, f"{prefix}_f1_scores.png")
    plt.tight_layout()
    plt.savefig(f1_path, dpi=160)
    plt.close()

    return txt_path, cm_path, f1_path
