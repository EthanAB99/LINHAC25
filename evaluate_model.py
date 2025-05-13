# ==========================================================
# evaluate_model.py – Model Evaluation: Accuracy, Confusion Matrix, Metrics + Charts
# ==========================================================

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, precision_recall_fscore_support
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from model_def import HockeyModel, HockeyDS, collate
from data_loader import X_test, y_test, event_vocab, type_vocab
from sklearn.calibration import calibration_curve
# Load model
model = HockeyModel(event_vocab_size=len(event_vocab)+1, type_vocab_size=len(type_vocab)+1)
model.load_state_dict(torch.load("zone_exit_lstm_state_dict.pt"), strict=False)
model.eval()

# DataLoader for test data
test_loader = DataLoader(HockeyDS(X_test, y_test), batch_size=32, shuffle=False, collate_fn=collate)

# Evaluation metrics
y_true = []
y_pred = []
y_prob = []
criterion = torch.nn.CrossEntropyLoss()
t_loss = t_corr = t_tot = 0

with torch.no_grad():
    for ev, tp, num, lbl, L, _ in test_loader:
        out = model(ev, tp, num, L)
        loss = criterion(out, lbl)
        b = lbl.size(0)
        t_tot += b
        t_loss += loss.item() * b
        t_corr += (out.argmax(1) == lbl).sum().item()
        y_true += lbl.cpu().tolist()
        y_pred += out.argmax(1).cpu().tolist()
        y_prob += torch.softmax(out, dim=1).cpu().tolist()

# Core stats
labels = ["Controlled", "Uncontrolled", "Fail"]
y_true_np = np.array(y_true)
y_prob_np = np.array(y_prob)
y_pred_np = np.array(y_pred)

roc_auc = roc_auc_score(y_true_np, y_prob_np, multi_class="ovr")
brier = np.mean(np.sum((y_prob_np - np.eye(3)[y_true_np])**2, axis=1))
acc = accuracy_score(y_true_np, y_pred_np)
cross_entropy = t_loss / t_tot

# Printout
print(f"\n[Test Eval] Cross-Entropy Loss: {cross_entropy:.4f}")
print(f"[Test Eval] Accuracy          : {acc:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4, target_names=labels))
print("=== EXTRA STATS ===")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"Brier Score    : {brier:.4f}")

# =========================
# 📊 CHARTS
# =========================

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Metrics Bar Chart
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])
report_df = pd.DataFrame({
    "Class": labels,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1
})
report_df.set_index("Class")[["Precision", "Recall", "F1-Score"]].plot(
    kind="bar", figsize=(8, 5), ylim=(0, 1), rot=0
)
plt.title("Classification Metrics by Class")
plt.ylabel("Score")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("classification_report.png")
plt.show()

# Prediction Distribution Chart (optional)
pred_series = pd.Series(y_pred).map(dict(enumerate(labels)))
plt.figure(figsize=(6, 4))
sns.countplot(x=pred_series, order=labels)
plt.title("Prediction Distribution")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("prediction_distribution.png")
plt.show()

from sklearn.calibration import calibration_curve

# Reliability chart (calibration curve) for class 0 (Controlled) as an example
true_class = 0
probs = y_prob_np[:, true_class]
true_labels_binary = (y_true_np == true_class).astype(int)

# Compute calibration curve
prob_true, prob_pred = calibration_curve(true_labels_binary, probs, n_bins=10, strategy='uniform')

from sklearn.calibration import calibration_curve

# Reliability diagram (calibration curve) for all 3 classes
plt.figure(figsize=(7, 6))

for i, name in enumerate(labels):
    # Binary true/false for current class
    true_binary = (y_true_np == i).astype(int)
    probs = y_prob_np[:, i]

    # Get calibration curve
    prob_true, prob_pred = calibration_curve(true_binary, probs, n_bins=10, strategy='uniform')
    
    # Plot per class
    plt.plot(prob_pred, prob_true, marker='o', label=f"{name}")

# Reference line for perfect calibration
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

# Chart formatting
plt.title("Reliability Curve (All Classes)")
plt.xlabel("Predicted Probability")
plt.ylabel("Empirical Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reliability_curve.png")
plt.show()

