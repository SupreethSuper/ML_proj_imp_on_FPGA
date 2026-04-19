"""
compare_tf_xgboost.py
=====================
Head-to-head comparison of the TensorFlow DNN and XGBoost classifiers
for S&P 500 Buy / Hold / Sell prediction.

Trains BOTH models on the same data split, computes identical metrics,
and produces comparative visualisation charts.

Outputs
-------
- Console: side-by-side metrics table
- Saved PNGs:
    CMP_Metrics_BarChart.png            – grouped bar chart of key metrics
    CMP_Confusion_Matrices.png          – side-by-side confusion matrices
    CMP_Confidence_BoxPlot.png          – box-plot of confidence distributions
    CMP_Feature_Importance_Compared.png – feature importance overlay
    CMP_Per_Class_F1.png                – per-class F1 comparison
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, precision_score, recall_score
)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import xgboost as xgb

SAVE_DIR = os.path.dirname(__file__)

# ============================================================
# 1. Shared Data Loading
# ============================================================
DATA_PATH = os.path.join(SAVE_DIR, "..", "datasets", "SP500V3.xlsx")
SHEET     = "SP500"

print("=" * 60)
print("  COMPARISON: TensorFlow DNN  vs  XGBoost")
print("=" * 60)
print(f"\nReading sheet '{SHEET}' from '{DATA_PATH}' ...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")

TARGET_COL = "Target: Buy/hold/sell = 1/0/-1 0.3% margin"
df = df.dropna(subset=[TARGET_COL])

exclude = {"observation_date",
           "Target: Raw next-day return",
           "Target: Up or Down - 1 next day UP, 0 next day down/flat",
           TARGET_COL}
feature_cols = [c for c in df.columns if c not in exclude]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feature_cols)

X = df[feature_cols].values
Y = df[TARGET_COL].astype(int).values
Y_shifted = Y + 1

CLASS_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y_shifted[:split_idx], Y_shifted[split_idx:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# 2. Train TensorFlow DNN
# ============================================================
print("\n--- Training TensorFlow DNN ---")

tf_model = keras.Sequential([
    layers.Input(shape=(X_train_sc.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax")
])
tf_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])

tf_history = tf_model.fit(
    X_train_sc, y_train,
    validation_split=0.15, epochs=150, batch_size=64,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                 restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=5, min_lr=1e-6)
    ],
    verbose=0
)

tf_proba = tf_model.predict(X_test_sc, verbose=0)
tf_pred  = np.argmax(tf_proba, axis=1)
tf_conf  = np.max(tf_proba, axis=1)

# ============================================================
# 3. Train XGBoost
# ============================================================
print("--- Training XGBoost ---")

counter = Counter(y_train)
total   = len(y_train)
sw_dict = {c: total / (len(counter) * cnt) for c, cnt in counter.items()}
sw      = np.array([sw_dict[y] for y in y_train])

eval_split = int(len(X_train_sc) * 0.85)

xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, gamma=1,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", use_label_encoder=False,
    random_state=42, early_stopping_rounds=30, verbosity=0
)
xgb_model.fit(
    X_train_sc[:eval_split], y_train[:eval_split],
    sample_weight=sw[:eval_split],
    eval_set=[(X_train_sc[eval_split:], y_train[eval_split:])],
    verbose=False
)

xgb_proba = xgb_model.predict_proba(X_test_sc)
xgb_pred  = np.argmax(xgb_proba, axis=1)
xgb_conf  = np.max(xgb_proba, axis=1)

# ============================================================
# 4. Compute All Metrics
# ============================================================
y_test_orig  = y_test - 1
tf_pred_orig = tf_pred - 1
xgb_pred_orig = xgb_pred - 1

def compute_metrics(y_true, y_pred, confidence):
    return {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro F1":          f1_score(y_true, y_pred, average="macro"),
        "Macro Precision":   precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall":      recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Mean Confidence":   confidence.mean(),
        "Median Confidence": np.median(confidence),
    }

tf_metrics  = compute_metrics(y_test_orig, tf_pred_orig,  tf_conf)
xgb_metrics = compute_metrics(y_test_orig, xgb_pred_orig, xgb_conf)

# Print side-by-side
print("\n" + "=" * 60)
print(f"{'Metric':<25s} {'TensorFlow':>12s} {'XGBoost':>12s} {'Winner':>10s}")
print("-" * 60)
for key in tf_metrics:
    tv = tf_metrics[key]
    xv = xgb_metrics[key]
    winner = "TF" if tv > xv else ("XGB" if xv > tv else "TIE")
    print(f"{key:<25s} {tv:>12.4f} {xv:>12.4f} {winner:>10s}")
print("=" * 60)

print("\n--- TensorFlow Classification Report ---")
print(classification_report(y_test_orig, tf_pred_orig, target_names=CLASS_NAMES))

print("--- XGBoost Classification Report ---")
print(classification_report(y_test_orig, xgb_pred_orig, target_names=CLASS_NAMES))

# ============================================================
# 5. Chart 1 – Grouped Bar Chart of Key Metrics
# ============================================================
metric_keys = ["Accuracy", "Balanced Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
tf_vals  = [tf_metrics[k]  for k in metric_keys]
xgb_vals = [xgb_metrics[k] for k in metric_keys]

x = np.arange(len(metric_keys))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, tf_vals,  width, label="TensorFlow DNN", color="steelblue")
bars2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost",        color="teal")

ax.set_ylabel("Score"); ax.set_title("TensorFlow vs XGBoost – Key Metrics Comparison", fontsize=15)
ax.set_xticks(x); ax.set_xticklabels(metric_keys, fontsize=11)
ax.set_ylim(0, 1); ax.legend(fontsize=12)

for bar in bars1 + bars2:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "CMP_Metrics_BarChart.png"), dpi=300)
print("\nSaved CMP_Metrics_BarChart.png")

# ============================================================
# 6. Chart 2 – Side-by-side Confusion Matrices
# ============================================================
cm_tf  = confusion_matrix(y_test_orig, tf_pred_orig,  labels=[-1, 0, 1])
cm_xgb = confusion_matrix(y_test_orig, xgb_pred_orig, labels=[-1, 0, 1])

cm_tf_n  = cm_tf.astype("float")  / cm_tf.sum(axis=1,  keepdims=True)
cm_xgb_n = cm_xgb.astype("float") / cm_xgb.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm_tf_n, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title("TensorFlow DNN", fontsize=13)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

sns.heatmap(cm_xgb_n, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title("XGBoost", fontsize=13)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.suptitle("Normalized Confusion Matrices – Side by Side", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "CMP_Confusion_Matrices.png"), dpi=300, bbox_inches="tight")
print("Saved CMP_Confusion_Matrices.png")

# ============================================================
# 7. Chart 3 – Confidence Box Plot
# ============================================================
conf_df = pd.DataFrame({
    "Confidence": np.concatenate([tf_conf, xgb_conf]),
    "Model":      ["TensorFlow"] * len(tf_conf) + ["XGBoost"] * len(xgb_conf)
})

plt.figure(figsize=(8, 6))
sns.boxplot(x="Model", y="Confidence", data=conf_df, palette=["steelblue", "teal"])
plt.title("Prediction Confidence Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "CMP_Confidence_BoxPlot.png"), dpi=300)
print("Saved CMP_Confidence_BoxPlot.png")

# ============================================================
# 8. Chart 4 – Feature Importance Comparison
# ============================================================
# XGBoost native importance
xgb_imp = xgb_model.feature_importances_

# TF permutation importance
from sklearn.inspection import permutation_importance

class KerasClassifierWrapper:
    def __init__(self, keras_model):
        self._model = keras_model
    def predict(self, X):
        return np.argmax(self._model.predict(X, verbose=0), axis=1)
    def fit(self, *a, **kw):
        pass

perm = permutation_importance(KerasClassifierWrapper(tf_model), X_test_sc, y_test,
                               n_repeats=10, random_state=42,
                               scoring="accuracy")
tf_imp = perm.importances_mean
# Normalise both to [0,1] for visual comparison
tf_imp_norm  = tf_imp  / (tf_imp.max()  + 1e-10)
xgb_imp_norm = xgb_imp / (xgb_imp.max() + 1e-10)

x = np.arange(len(feature_cols))
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - 0.2, tf_imp_norm,  0.4, label="TensorFlow (permutation)", color="steelblue", alpha=0.8)
ax.bar(x + 0.2, xgb_imp_norm, 0.4, label="XGBoost (gain)",           color="teal",      alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(feature_cols, rotation=45, ha="right")
ax.set_ylabel("Normalised Importance")
ax.set_title("Feature Importance Comparison (normalised)", fontsize=15)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "CMP_Feature_Importance_Compared.png"), dpi=300)
print("Saved CMP_Feature_Importance_Compared.png")

# ============================================================
# 9. Chart 5 – Per-class F1 Comparison
# ============================================================
from sklearn.metrics import f1_score as f1

tf_f1_per  = f1(y_test_orig, tf_pred_orig,  labels=[-1, 0, 1], average=None)
xgb_f1_per = f1(y_test_orig, xgb_pred_orig, labels=[-1, 0, 1], average=None)

x = np.arange(3)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.18, tf_f1_per,  0.35, label="TensorFlow", color="steelblue")
ax.bar(x + 0.18, xgb_f1_per, 0.35, label="XGBoost",    color="teal")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=12)
ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1)
ax.set_title("Per-Class F1 Score Comparison", fontsize=14)
ax.legend(fontsize=11)

for i in range(3):
    ax.text(i - 0.18, tf_f1_per[i]  + 0.02, f"{tf_f1_per[i]:.3f}",  ha="center", fontsize=9)
    ax.text(i + 0.18, xgb_f1_per[i] + 0.02, f"{xgb_f1_per[i]:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "CMP_Per_Class_F1.png"), dpi=300)
print("Saved CMP_Per_Class_F1.png")

# ============================================================
# 10. Summary Verdict
# ============================================================
tf_total  = sum(1 for k in metric_keys if tf_metrics[k] > xgb_metrics[k])
xgb_total = sum(1 for k in metric_keys if xgb_metrics[k] > tf_metrics[k])

print("\n" + "=" * 60)
print("  FINAL VERDICT")
print("=" * 60)
if xgb_total > tf_total:
    print(f"  XGBoost wins on {xgb_total}/{len(metric_keys)} key metrics.")
    print("  Recommendation: Use XGBoost for this dataset.")
elif tf_total > xgb_total:
    print(f"  TensorFlow DNN wins on {tf_total}/{len(metric_keys)} key metrics.")
    print("  Recommendation: Use TensorFlow DNN for this dataset.")
else:
    print("  Both models are tied on key metrics.")
    print("  Recommendation: Use XGBoost for faster inference / simpler deployment.")
print("=" * 60)

print("\n✓ compare_tf_xgboost.py completed successfully.")
