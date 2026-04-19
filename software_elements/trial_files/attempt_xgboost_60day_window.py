"""
attempt_xgboost_60day_window.py
==============================
S&P 500 Buy / Hold / Sell classification using XGBoost with configurable date window.

Date range: Configurable via command-line arguments
  - Default: 04-04-2016 to 27-06-2016 (60 days)
  
Usage examples:
  python attempt_xgboost_60day_window.py
  python attempt_xgboost_60day_window.py --start-date 2016-05-01 --end-date 2016-07-31
  python attempt_xgboost_60day_window.py --start-date 2016-04-04 --days 90 --epochs 300
  python attempt_xgboost_60day_window.py --epochs 1000

Same model logic as attempt_xgboost2.py, but with configurable date ranges.

Outputs
-------
- Console: accuracy, balanced accuracy, macro-F1, classification report, confusion matrix
- Saved PNGs:
    XGB_60day_Confusion_Matrix.png        – normalized confusion matrix heat-map
    XGB_60day_Feature_Importance.png      – gain-based feature importance
    XGB_60day_Confidence_Distribution.png – per-class softmax confidence histogram
    XGB_60day_Tree_Visualization.png      – first tree structure
"""

import os
import warnings
import argparse
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import xgboost as xgb

# ============================================================
# 0. Command-line Arguments for Date Range
# ============================================================
parser = argparse.ArgumentParser(
    description="XGBoost classifier for S&P 500 with configurable date window"
)
parser.add_argument(
    "--start-date",
    type=str,
    default="2016-04-04",
    help="Start date (YYYY-MM-DD). Default: 2016-04-04"
)
parser.add_argument(
    "--end-date",
    type=str,
    default="2016-06-27",
    help="End date (YYYY-MM-DD). Default: 2016-06-27"
)
parser.add_argument(
    "--days",
    type=int,
    default=None,
    help="Alternative: specify number of days from start-date instead of end-date"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=500,
    help="Number of boosting rounds (estimators). Default: 500"
)

args = parser.parse_args()

# ============================================================
# 1. Load & Prepare Data (configurable date window)
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "SP500V3.xlsx")
SHEET = "SP500"

print(f"Reading sheet '{SHEET}' from '{DATA_PATH}' ...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")

# Parse start date and calculate end date
date_col = "observation_date"
df[date_col] = pd.to_datetime(df[date_col])
start_date = pd.to_datetime(args.start_date)

# If --days is specified, calculate end_date from start_date + days
if args.days is not None:
    end_date = start_date + pd.Timedelta(days=args.days)
else:
    end_date = pd.to_datetime(args.end_date)

days_count = (end_date - start_date).days

df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
df = df.sort_values(by=date_col).reset_index(drop=True)

print(f"Filtered to {days_count}-day window: {len(df)} rows")
print(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}\n")

TARGET_COL = "Target: Buy/hold/sell = 1/0/-1 0.3% margin"
df = df.dropna(subset=[TARGET_COL])

# Feature columns (same as attempt_xgboost2.py for consistency)
exclude = {
    "observation_date",
    "Target: Raw next-day return",
    "Target: Up or Down - 1 next day UP, 0 next day down/flat",
    TARGET_COL
}
feature_cols = [c for c in df.columns if c not in exclude]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feature_cols)

X = df[feature_cols].values
Y = df[TARGET_COL].astype(int).values   # -1, 0, 1
Y_shifted = Y + 1                       # 0, 1, 2 for XGBoost

CLASS_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]

print(f"Final dataset: {len(X)} rows | {len(feature_cols)} features")
print(f"XGBoost epochs (n_estimators): {args.epochs}\n")

# ============================================================
# 2. Chronological Train / Test Split (80 / 20)
# ============================================================
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y_shifted[:split_idx], Y_shifted[split_idx:]

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ============================================================
# 3. Build & Train XGBoost Classifier
# ============================================================
counter = Counter(y_train)
total = len(y_train)
sample_weight_dict = {c: total / (len(counter) * cnt) for c, cnt in counter.items()}
sample_weights = np.array([sample_weight_dict[y] for y in y_train])

xgb_model = xgb.XGBClassifier(
    n_estimators=args.epochs,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.01,
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
y_proba = xgb_model.predict_proba(X_test_sc)
y_pred = np.argmax(y_proba, axis=1)
confidence = np.max(y_proba, axis=1)

y_test_orig = y_test - 1
y_pred_orig = y_pred - 1

# ============================================================
# 5. Evaluation Metrics
# ============================================================
acc = accuracy_score(y_test_orig, y_pred_orig)
bal_acc = balanced_accuracy_score(y_test_orig, y_pred_orig)
f1_mac = f1_score(y_test_orig, y_pred_orig, average="macro")

print("\n" + "=" * 50)
print(f"--- XGBoost Model Evaluation ({days_count}-day window) ---")
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

# Avoid divide-by-zero if any row sums to zero
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm.astype(float), row_sums, where=(row_sums != 0))
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Greens")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalized Value")

ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted",
    ylabel="Actual",
    title=f"XGBoost ({days_count}-day) – Normalized Confusion Matrix"
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(
            j, i, f"{cm_norm[i, j]:.2f}",
            ha="center", va="center", color="black"
        )

fig.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "XGB_60day_Confusion_Matrix.png"),
    dpi=300
)
print("\nSaved XGB_60day_Confusion_Matrix.png")
plt.close(fig)

# ============================================================
# 7. Visualization 2 – Feature Importance
# ============================================================
importances = xgb_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(feature_cols)), importances[sorted_idx])
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels(
    [feature_cols[i] for i in sorted_idx],
    rotation=45,
    ha="right"
)
ax.set_title("XGBoost ({}-day) – Feature Importance (Gain)".format(days_count), fontsize=16)
ax.set_xlabel("Feature")
ax.set_ylabel("Importance Score")

fig.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "XGB_60day_Feature_Importance.png"),
    dpi=300
)
print("Saved XGB_60day_Feature_Importance.png")
plt.close(fig)

# ============================================================
# 8. Visualization 3 – Confidence Distribution per Class
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = y_pred == cls_idx
    if mask.sum() > 0:
        ax.hist(confidence[mask], bins=30, alpha=0.5, label=cls_name)

ax.set_title("XGBoost ({}-day) – Prediction Confidence Distribution".format(days_count), fontsize=14)
ax.set_xlabel("Confidence (max softmax probability)")
ax.set_ylabel("Count")
ax.legend()

fig.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "XGB_60day_Confidence_Distribution.png"),
    dpi=300
)
print("Saved XGB_60day_Confidence_Distribution.png")
plt.close(fig)

# ============================================================
# 9. Visualization 4 – Single Tree Plot
# ============================================================
try:
    fig, ax = plt.subplots(figsize=(24, 12))
    xgb.plot_tree(xgb_model, num_trees=0, ax=ax, rankdir="LR")
    ax.set_title("XGBoost ({}-day) – First Estimator Tree Structure".format(days_count), fontsize=16)
    fig.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "XGB_60day_Tree_Visualization.png"),
        dpi=200,
        bbox_inches="tight"
    )
    print("Saved XGB_60day_Tree_Visualization.png")
    plt.close(fig)

except Exception as e:
    print(f"[INFO] xgb.plot_tree unavailable ({type(e).__name__}: {e})")
    print("[INFO] Falling back to text-based tree dump rendered as matplotlib image.")

    booster = xgb_model.get_booster()
    tree_dump = booster.get_dump(with_stats=True)[0]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.text(
        0.01, 0.99, tree_dump,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    ax.set_axis_off()
    ax.set_title("XGBoost ({}-day) – First Estimator Tree (text dump)".format(days_count), fontsize=16)
    fig.tight_layout()

    plt.savefig(
        os.path.join(os.path.dirname(__file__), "XGB_60day_Tree_Visualization.png"),
        dpi=200,
        bbox_inches="tight"
    )
    print("Saved XGB_60day_Tree_Visualization.png (text fallback)")
    plt.close(fig)

print("\n✓ attempt_xgboost_60day_window.py completed successfully.")
