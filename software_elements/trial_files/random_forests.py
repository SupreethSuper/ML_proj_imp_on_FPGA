"""
Random Forest Binary Classification for SP500 Up/Down Prediction
=================================================================
Dataset : SP500V3 - Copy.xlsx  (sheet "SP500")
Target  : Up or Down - 1 next day UP, 0 next day down/flat
Features: All available numeric indicators + NLP_predict + news_article_count
Lookback: 10-day sliding window (stride 1), flattened into feature vectors
Split   : 70/30 random split (NOT chronological)
Output  : Confidence-score plots via matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# -- reproducibility ---------------------------------------------------
SEED = 42
np.random.seed(SEED)

# -- paths -------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets", "SP500V3 - Copy.xlsx")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "random_forest_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- hyperparameters ---------------------------------------------------
LOOKBACK      = 10       # 10-day sliding window
TEST_RATIO    = 0.3      # 70/30 split
N_ESTIMATORS  = 300      # number of trees
MAX_DEPTH     = 15
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF  = 5

# =====================================================================
# 1.  LOAD & PREPROCESS
# =====================================================================
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
df = pd.read_excel(DATASET_PATH, sheet_name="SP500", engine="openpyxl")
print(f"Raw shape: {df.shape}")

NUMERIC_FEATURES = [
    "SP500",
    "SP500 1 day return",
    "SP500 3 day momentum",
    "VIX",
    "VIX 1 day change",
    "DGS10",
    "DGS10 1 day change",
    "T10Y3M ",               # trailing space in original header
    "BAMLC0A4CBBB",
    "CPIAUCSL",
    "DFF",
    "DGS2",
    "DGS2 1 day change",
    "Closing Value - CRUDE OIL",
    "news_article_count",
]

TARGET_COL = "Target: Up or Down - 1 next day UP, 0 next day down/flat"

# encode NLP_predict (buy=1, sell=0)
df["NLP_encoded"] = (df["NLP_predict"] == "buy").astype(int)
NUMERIC_FEATURES.append("NLP_encoded")

# drop rows where target is NaN
df = df.dropna(subset=[TARGET_COL])
df["target"] = df[TARGET_COL].astype(int)
print(f"After dropping NaN targets: {df.shape}")

# fill missing numeric values
for col in NUMERIC_FEATURES:
    df[col] = df[col].ffill().bfill()
df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

print(f"Target distribution:")
print(f"  Up   (1): {(df['target'] == 1).sum()}")
print(f"  Down (0): {(df['target'] == 0).sum()}")
print(f"Features used ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")

dates = df["observation_date"].values

# =====================================================================
# 2.  SCALE
# =====================================================================
scaler = MinMaxScaler()
feature_data = scaler.fit_transform(df[NUMERIC_FEATURES].values)
target_data  = df["target"].values

# =====================================================================
# 3.  CREATE SLIDING WINDOW SEQUENCES (flattened for RF)
# =====================================================================
# 10-day lookback, slide 1 day at a time
# Each sample = 10 consecutive days flattened into a single feature vector
# e.g. 10 days x 16 features = 160 features per sample

def create_sliding_windows(features, targets, dates_arr, lookback):
    X, y, d = [], [], []
    for i in range(lookback, len(features)):
        # flatten the lookback window into a single row
        window = features[i - lookback : i].flatten()
        X.append(window)
        y.append(targets[i])
        d.append(dates_arr[i])
    return np.array(X), np.array(y), np.array(d)

print(f"\nCreating {LOOKBACK}-day sliding windows (stride=1)...")
X, y, sample_dates = create_sliding_windows(feature_data, target_data, dates, LOOKBACK)

# build feature names for the flattened window
feature_names = []
for day in range(LOOKBACK):
    for feat in NUMERIC_FEATURES:
        feature_names.append(f"d-{LOOKBACK - day}_{feat}")

print(f"Sliding window dataset - X: {X.shape}, y: {y.shape}")
print(f"Each sample has {X.shape[1]} features "
      f"({LOOKBACK} days x {len(NUMERIC_FEATURES)} features)")

# =====================================================================
# 4.  RANDOM 70/30 TRAIN-TEST SPLIT
# =====================================================================
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, sample_dates,
    test_size=TEST_RATIO,
    random_state=SEED,
    stratify=y,             # maintain class balance
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}  (random split)")
print(f"Train dist: Up={int(y_train.sum())}, Down={int((y_train == 0).sum())}")
print(f"Test  dist: Up={int(y_test.sum())},  Down={int((y_test == 0).sum())}")

# =====================================================================
# 5.  BUILD & TRAIN RANDOM FOREST
# =====================================================================
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced",    # handle class imbalance
    random_state=SEED,
    n_jobs=-1,                  # use all CPU cores
    verbose=1,
)

rf.fit(X_train, y_train)
print("Training complete.")

# =====================================================================
# 6.  EVALUATE
# =====================================================================
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

# confidence scores: probability of class 1 (Up)
y_conf_train = rf.predict_proba(X_train)[:, 1]
y_conf_test  = rf.predict_proba(X_test)[:, 1]
y_pred_test  = rf.predict(X_test)
y_pred_train = rf.predict(X_train)

acc  = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec  = recall_score(y_test, y_pred_test, zero_division=0)
f1   = f1_score(y_test, y_pred_test, zero_division=0)
auc  = roc_auc_score(y_test, y_conf_test)

train_acc = accuracy_score(y_train, y_pred_train)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test  Accuracy: {acc:.4f}")
print(f"Precision     : {prec:.4f}")
print(f"Recall        : {rec:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC-AUC       : {auc:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_test, y_pred_test, target_names=['Down (0)', 'Up (1)'])}")

# =====================================================================
# 7.  CONFIDENCE SCORE PLOTS
# =====================================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

# ---- 7a. Confidence scores over time (test set, sorted by date) ----
# sort test samples by date for temporal visualization
sort_idx = np.argsort(dates_test)
sorted_dates = pd.to_datetime(dates_test[sort_idx])
sorted_conf  = y_conf_test[sort_idx]
sorted_y     = y_test[sort_idx]

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(sorted_dates, sorted_conf, linewidth=0.8, color="#2196F3",
        alpha=0.8, label="Confidence Score (P(Up))")
ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1,
           label="Decision Threshold (0.5)")
ax.fill_between(sorted_dates, 0.5, sorted_conf,
                where=(sorted_conf >= 0.5), alpha=0.15, color="green",
                label="Up Region")
ax.fill_between(sorted_dates, 0.5, sorted_conf,
                where=(sorted_conf < 0.5), alpha=0.15, color="red",
                label="Down Region")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Confidence Score", fontsize=12)
ax.set_title("Random Forest - Confidence Scores Over Time (Test Set)", fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fname = "RF_confidence_timeline.png"
fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7b. Confidence distribution histogram ----
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(y_conf_test[y_test == 1], bins=50, alpha=0.6, color="green",
         label="Actual Up (1)", density=True)
ax2.hist(y_conf_test[y_test == 0], bins=50, alpha=0.6, color="red",
         label="Actual Down (0)", density=True)
ax2.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5,
            label="Threshold")
ax2.set_xlabel("Confidence Score", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Confidence Score Distribution by True Class", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fname = "RF_confidence_distribution.png"
fig2.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7c. ROC Curve ----
fpr, tpr, _ = roc_curve(y_test, y_conf_test)
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.plot(fpr, tpr, color="#2196F3", linewidth=2,
         label=f"ROC Curve (AUC = {auc:.4f})")
ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
ax3.set_xlabel("False Positive Rate", fontsize=12)
ax3.set_ylabel("True Positive Rate", fontsize=12)
ax3.set_title("ROC Curve - Random Forest", fontsize=14)
ax3.legend(loc="lower right", fontsize=12)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fname = "RF_roc_curve.png"
fig3.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7d. Feature importance (top 30) ----
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

top_n = 30
fig4, ax4 = plt.subplots(figsize=(12, 8))
top_feats = feat_imp_df.head(top_n)
colors_bar = []
for f in top_feats["feature"]:
    day_part = f.split("_")[0]  # e.g. "d-1", "d-10"
    day_num = int(day_part.split("-")[1])
    # color gradient: more recent days = darker
    intensity = 1 - (day_num - 1) / LOOKBACK
    colors_bar.append(plt.cm.Blues(0.3 + 0.7 * intensity))

ax4.barh(range(top_n - 1, -1, -1), top_feats["importance"].values,
         color=colors_bar)
ax4.set_yticks(range(top_n - 1, -1, -1))
ax4.set_yticklabels(top_feats["feature"].values, fontsize=9)
ax4.set_xlabel("Importance", fontsize=12)
ax4.set_title(f"Top {top_n} Feature Importances (darker = more recent day)", fontsize=14)
ax4.grid(True, alpha=0.3, axis="x")
fig4.tight_layout()
fname = "RF_feature_importance.png"
fig4.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7e. Confusion matrix heatmap ----
fig5, ax5 = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred_test)
im = ax5.imshow(cm, interpolation="nearest", cmap="Blues")
fig5.colorbar(im, ax=ax5)
class_labels = ["Down (0)", "Up (1)"]
ax5.set(xticks=range(2), yticks=range(2),
        xticklabels=class_labels, yticklabels=class_labels)
ax5.set_xlabel("Predicted", fontsize=12)
ax5.set_ylabel("Actual", fontsize=12)
ax5.set_title("Confusion Matrix - Random Forest", fontsize=14)
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax5.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color=color, fontsize=16, fontweight="bold")
fig5.tight_layout()
fname = "RF_confusion_matrix.png"
fig5.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7f. Confidence scatter by true class ----
fig6, ax6 = plt.subplots(figsize=(10, 6))
colors_scatter = ["red" if yy == 0 else "green" for yy in y_test]
ax6.scatter(range(len(y_conf_test)), y_conf_test, c=colors_scatter,
            alpha=0.4, s=10, edgecolors="none")
ax6.axhline(y=0.5, color="black", linestyle="--", linewidth=1.5)
ax6.set_xlabel("Test Sample Index", fontsize=12)
ax6.set_ylabel("Confidence Score", fontsize=12)
ax6.set_title("Confidence Scores - Green=Up(actual), Red=Down(actual)", fontsize=14)
ax6.grid(True, alpha=0.3)
fig6.tight_layout()
fname = "RF_confidence_scatter.png"
fig6.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7g. Train vs Test Accuracy bar chart ----
fig7, ax7 = plt.subplots(figsize=(8, 6))
bars = ax7.bar(["Train Accuracy", "Test Accuracy"],
               [train_acc, acc],
               color=["#2196F3", "#E53935"], width=0.5)
for bar, val in zip(bars, [train_acc, acc]):
    ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax7.set_ylabel("Accuracy", fontsize=12)
ax7.set_title("Train vs Test Accuracy - Random Forest", fontsize=14)
ax7.set_ylim(0, 1.0)
ax7.grid(True, alpha=0.3, axis="y")
fig7.tight_layout()
fname = "RF_train_vs_test_accuracy.png"
fig7.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7h. Number of trees vs OOB / accuracy (if OOB enabled) ----
# Train a range of forests to show accuracy vs n_estimators
print("\n  Computing accuracy vs number of trees...")
tree_counts = [10, 25, 50, 100, 150, 200, 250, 300]
train_accs_trees = []
test_accs_trees  = []

for n in tree_counts:
    rf_tmp = RandomForestClassifier(
        n_estimators=n,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_tmp.fit(X_train, y_train)
    train_accs_trees.append(accuracy_score(y_train, rf_tmp.predict(X_train)))
    test_accs_trees.append(accuracy_score(y_test, rf_tmp.predict(X_test)))

fig8, ax8 = plt.subplots(figsize=(10, 6))
ax8.plot(tree_counts, train_accs_trees, "o-", label="Train Accuracy",
         linewidth=1.5, color="#2196F3")
ax8.plot(tree_counts, test_accs_trees, "s-", label="Test Accuracy",
         linewidth=1.5, color="#E53935")
ax8.set_xlabel("Number of Trees", fontsize=12)
ax8.set_ylabel("Accuracy", fontsize=12)
ax8.set_title("Accuracy vs Number of Trees", fontsize=14)
ax8.legend(fontsize=11)
ax8.grid(True, alpha=0.3)
fig8.tight_layout()
fname = "RF_accuracy_vs_trees.png"
fig8.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# =====================================================================
# 8.  SAVE RESULTS
# =====================================================================
import joblib
joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest_model.joblib"))
print(f"\n  Model saved: random_forest_model.joblib")

results_df = pd.DataFrame({
    "date": dates_test,
    "confidence_score": y_conf_test,
    "predicted_label": y_pred_test,
    "actual_label": y_test.astype(int),
    "correct": (y_pred_test == y_test).astype(int),
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
print("  Test predictions saved: test_predictions.csv")

# save feature importances
feat_imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
print("  Feature importances saved: feature_importances.csv")

plt.close("all")
print("\n" + "=" * 60)
print("DONE - All outputs saved to:", OUTPUT_DIR)
print("=" * 60)
