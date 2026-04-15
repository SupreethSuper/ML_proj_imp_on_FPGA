"""
attempt_tensorflow1.py
======================
S&P 500 Buy / Hold / Sell classification using a TensorFlow (Keras) Deep Neural Network.

Reference dataset : SP500V3.xlsx  →  sheet "SP500"
Reference code    : project.py (Decision Tree baseline)

Outputs
-------
- Console: accuracy, balanced accuracy, macro-F1, classification report, confusion matrix
- Saved PNGs:
    TF_Training_History.png        – loss / accuracy curves over epochs
    TF_Confusion_Matrix.png        – normalized confusion matrix heat-map
    TF_Feature_Importance.png      – permutation-based feature importance
    TF_Confidence_Distribution.png – per-class softmax confidence histogram
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF info logs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ============================================================
# 1. Load & Prepare Data
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "SP500V3.xlsx")
SHEET     = "SP500"

print(f"Reading sheet '{SHEET}' from '{DATA_PATH}' ...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")

TARGET_COL = "Target: Buy/hold/sell = 1/0/-1 0.3% margin"
df = df.dropna(subset=[TARGET_COL])

# Feature columns (all numeric columns except targets and date)
exclude = {"observation_date",
           "Target: Raw next-day return",
           "Target: Up or Down - 1 next day UP, 0 next day down/flat",
           TARGET_COL}
feature_cols = [c for c in df.columns if c not in exclude]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feature_cols)

X = df[feature_cols].values
Y = df[TARGET_COL].astype(int).values          # -1, 0, 1
Y_shifted = Y + 1                               # map to 0, 1, 2 for keras

CLASS_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]

# ============================================================
# 2. Chronological Train / Test Split  (80 / 20)
# ============================================================
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y_shifted[:split_idx], Y_shifted[split_idx:]

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ============================================================
# 3. Build TensorFlow / Keras DNN
# ============================================================
def build_model(input_dim, num_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(X_train.shape[1])
model.summary()

# ============================================================
# 4. Train the Model
# ============================================================
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# ============================================================
# 5. Predictions & Confidence Scores
# ============================================================
y_proba = model.predict(X_test)                 # softmax probabilities
y_pred  = np.argmax(y_proba, axis=1)
confidence = np.max(y_proba, axis=1)            # max prob = confidence

# Map back to original labels for reporting
y_test_orig = y_test - 1
y_pred_orig = y_pred - 1

# ============================================================
# 6. Evaluation Metrics  (matching project.py style)
# ============================================================
acc     = accuracy_score(y_test_orig, y_pred_orig)
bal_acc = balanced_accuracy_score(y_test_orig, y_pred_orig)
f1_mac  = f1_score(y_test_orig, y_pred_orig, average="macro")

print("\n" + "=" * 50)
print("--- TensorFlow DNN Model Evaluation ---")
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
# 7. Visualization 1 – Training History
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss Over Epochs", fontsize=14)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[1].set_title("Accuracy Over Epochs", fontsize=14)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.suptitle("TensorFlow DNN – Training History", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "TF_Training_History.png"), dpi=300, bbox_inches="tight")
print("\nSaved TF_Training_History.png")

# ============================================================
# 8. Visualization 2 – Normalized Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test_orig, y_pred_orig, labels=[-1, 0, 1])
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("TensorFlow DNN – Normalized Confusion Matrix", fontsize=14)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "TF_Confusion_Matrix.png"), dpi=300)
print("Saved TF_Confusion_Matrix.png")

# ============================================================
# 9. Visualization 3 – Permutation Feature Importance
# ============================================================
from sklearn.inspection import permutation_importance

# Wrap the Keras model so .predict() returns class labels (required by sklearn)
class KerasClassifierWrapper:
    def __init__(self, keras_model):
        self._model = keras_model
    def predict(self, X):
        return np.argmax(self._model.predict(X, verbose=2), axis=1)
    def fit(self, *args, **kwargs):
        pass

perm = permutation_importance(KerasClassifierWrapper(model), X_test, y_test,
                               n_repeats=10, random_state=42,
                               scoring="accuracy")

sorted_idx = perm.importances_mean.argsort()[::-1]

plt.figure(figsize=(14, 6))
plt.bar(range(len(feature_cols)),
        perm.importances_mean[sorted_idx],
        yerr=perm.importances_std[sorted_idx],
        color="steelblue")
plt.xticks(range(len(feature_cols)),
           [feature_cols[i] for i in sorted_idx],
           rotation=45, ha="right")
plt.title("TensorFlow DNN – Permutation Feature Importance", fontsize=16)
plt.xlabel("Feature"); plt.ylabel("Mean Accuracy Decrease")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "TF_Feature_Importance.png"), dpi=300)
print("Saved TF_Feature_Importance.png")

# ============================================================
# 10. Visualization 4 – Confidence Distribution per Class
# ============================================================
plt.figure(figsize=(10, 6))
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = y_pred == cls_idx
    if mask.sum() > 0:
        plt.hist(confidence[mask], bins=30, alpha=0.5, label=cls_name)
plt.title("TensorFlow DNN – Prediction Confidence Distribution", fontsize=14)
plt.xlabel("Confidence (max softmax probability)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "TF_Confidence_Distribution.png"), dpi=300)
print("Saved TF_Confidence_Distribution.png")

print("\n✓ attempt_tensorflow1.py completed successfully.")
