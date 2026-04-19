"""
attempt_xgboost1.py
====================
S&P 500 Buy / Hold / Sell classification using XGBoost (Gradient Boosted Trees).

Rationale for choosing XGBoost as the "highest accuracy" method:
  - XGBoost consistently wins tabular-data benchmarks (Kaggle, academic studies).
  - Handles class imbalance, missing values, and non-linear relationships natively.
  - Built-in regularization prevents overfitting on small financial datasets.
  - Much faster training than deep-learning on datasets of this size (~2 500 rows).

Reference dataset : SP500V3.xlsx  →  sheet "SP500"
Reference code    : project.py (Decision Tree baseline)

Outputs
-------
- Console: accuracy, balanced accuracy, macro-F1, classification report, confusion matrix
- Saved PNGs:
    XGB_Confusion_Matrix.png        – normalized confusion matrix heat-map
    XGB_Feature_Importance.png      – gain-based feature importance
    XGB_Confidence_Distribution.png – per-class softmax confidence histogram
    XGB_Tree_Visualization.png      – first tree structure (like project.py tree plot)
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

import xgboost as xgb

# ============================================================
# 1. Load & Prepare Data
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "SP500V3.xlsx")
SHEET     = "SP500"

print(f"Reading sheet '{SHEET}' from '{DATA_PATH}' ...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")

TARGET_COL = "Target: Buy/hold/sell = 1/0/-1 0.3% margin"
df = df.dropna(subset=[TARGET_COL])

# Feature columns (same as TF script for fair comparison)
exclude = {"observation_date",
           "Target: Raw next-day return",
           "Target: Up or Down - 1 next day UP, 0 next day down/flat",
           TARGET_COL}
feature_cols = [c for c in df.columns if c not in exclude]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feature_cols)

X = df[feature_cols].values
Y = df[TARGET_COL].astype(int).values           # -1, 0, 1
Y_shifted = Y + 1                                # 0, 1, 2  (XGBoost needs 0-indexed)

CLASS_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]

# ============================================================
# 2. Chronological Train / Test Split  (80 / 20)
# ============================================================
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y_shifted[:split_idx], Y_shifted[split_idx:]

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# Scaling (XGBoost doesn't strictly need it, but helps comparability)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# 3. Build & Train XGBoost Classifier
# ============================================================
# Compute class weights for imbalanced data
from collections import Counter
counter = Counter(y_train)
total   = len(y_train)
sample_weight_dict = {c: total / (len(counter) * cnt) for c, cnt in counter.items()}
sample_weights = np.array([sample_weight_dict[y] for y in y_train])

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    early_stopping_rounds=30,
    verbosity=2
)

# Use last 15% of training data as eval set
eval_split = int(len(X_train_sc) * 0.85)
xgb_model.fit(
    X_train_sc[:eval_split], y_train[:eval_split],
    sample_weight=sample_weights[:eval_split],
    eval_set=[(X_train_sc[eval_split:], y_train[eval_split:])],
    verbose=True
)

print(f"Best iteration: {xgb_model.best_iteration}")

# ============================================================
# 4. Predictions & Confidence Scores
# ============================================================
y_proba     = xgb_model.predict_proba(X_test_sc)
y_pred      = np.argmax(y_proba, axis=1)
confidence  = np.max(y_proba, axis=1)

y_test_orig = y_test - 1
y_pred_orig = y_pred - 1

# ============================================================
# 5. Evaluation Metrics (matching project.py style)
# ============================================================
acc     = accuracy_score(y_test_orig, y_pred_orig)
bal_acc = balanced_accuracy_score(y_test_orig, y_pred_orig)
f1_mac  = f1_score(y_test_orig, y_pred_orig, average="macro")

print("\n" + "=" * 50)
print("--- XGBoost Model Evaluation ---")
print("=" * 50)
print(f"Accuracy          : {acc:.4f}")
print(f"Balanced Accuracy : {bal_acc:.4f}")
print(f"Macro F1          : {f1_mac:.4f}")
print(f"\nMean Confidence   : {confidence.mean():.4f}")
print(f"Median Confidence : {np.median(confidence):.4f}")
print("\nClassification Report:\n",
      classification_report(y_test_orig, y_pred_orig, target_names=CLASS_NAMES))

print("Training label distribution:")
unique, counts = np.unique(y_train - 1, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {int(u):>2d}: {c}")

print("\nTest label distribution:")
unique, counts = np.unique(y_test_orig, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {int(u):>2d}: {c}")

# ============================================================
# 6. Visualization 1 – Normalized Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test_orig, y_pred_orig, labels=[-1, 0, 1])
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("XGBoost – Normalized Confusion Matrix", fontsize=14)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "XGB_Confusion_Matrix.png"), dpi=300)
print("\nSaved XGB_Confusion_Matrix.png")

# ============================================================
# 7. Visualization 2 – Feature Importance (Gain)
# ============================================================
importances = xgb_model.feature_importances_

plt.figure(figsize=(14, 6))
sorted_idx = np.argsort(importances)[::-1]
plt.bar(range(len(feature_cols)),
        importances[sorted_idx],
        color="teal")
plt.xticks(range(len(feature_cols)),
           [feature_cols[i] for i in sorted_idx],
           rotation=45, ha="right")
plt.title("XGBoost – Feature Importance (Gain)", fontsize=16)
plt.xlabel("Feature"); plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "XGB_Feature_Importance.png"), dpi=300)
print("Saved XGB_Feature_Importance.png")

# ============================================================
# 8. Visualization 3 – Confidence Distribution per Class
# ============================================================
plt.figure(figsize=(10, 6))
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = y_pred == cls_idx
    if mask.sum() > 0:
        plt.hist(confidence[mask], bins=30, alpha=0.5, label=cls_name)
plt.title("XGBoost – Prediction Confidence Distribution", fontsize=14)
plt.xlabel("Confidence (max softmax probability)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "XGB_Confidence_Distribution.png"), dpi=300)
print("Saved XGB_Confidence_Distribution.png")

# ============================================================
# 9. Visualization 4 – Single Tree Plot (like project.py)
# ============================================================
# xgb.plot_tree requires the Graphviz system binary ('dot').
# If it's not installed, fall back to a text-based dump rendered as a figure.
try:
    fig, ax = plt.subplots(figsize=(24, 12))
    xgb.plot_tree(xgb_model, num_trees=0, ax=ax, rankdir="LR")
    plt.title("XGBoost – First Estimator Tree Structure", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "XGB_Tree_Visualization.png"), dpi=200, bbox_inches="tight")
    print("Saved XGB_Tree_Visualization.png")
except Exception as e:
    print(f"[INFO] xgb.plot_tree unavailable ({type(e).__name__}: {e})")
    print("[INFO] Falling back to text-based tree dump rendered as image.")
    booster = xgb_model.get_booster()
    tree_dump = booster.get_dump(with_stats=True)[0]
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.text(0.01, 0.99, tree_dump, transform=ax.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_axis_off()
    ax.set_title("XGBoost – First Estimator Tree (text dump)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "XGB_Tree_Visualization.png"), dpi=200, bbox_inches="tight")
    print("Saved XGB_Tree_Visualization.png (text fallback)")

print("\n✓ attempt_xgboost1.py completed successfully.")
