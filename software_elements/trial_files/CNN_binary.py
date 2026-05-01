"""
CNN Binary Classification for SP500 Buy/Sell Prediction
========================================================
Dataset : SP500V3 - Copy.xlsx  (sheet "SP500")
Target  : Buy/hold/sell column -> binary (Buy=1, else=0)
Features: All numeric indicators + NLP_predict + news_article_count
Sequence: 30-day lookback window (treated as 1D signal for Conv1D)
Output  : Confidence-score plots via matplotlib

Architecture:
  Conv1D (16 features x 30 timesteps) -> Conv1D -> Conv1D -> GlobalAvgPool -> FC -> Sigmoid
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# -- reproducibility ---------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- paths -------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets", "SP500V3 - Copy.xlsx")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "cnn_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- hyperparameters ---------------------------------------------------
SEQ_LEN       = 30
TEST_RATIO    = 0.3
EPOCHS        = 100
BATCH_SIZE    = 32
DROPOUT_RATE  = 0.3
LEARNING_RATE = 0.001
PATIENCE      = 30

# CNN-specific
CONV1_FILTERS = 64
CONV2_FILTERS = 128
CONV3_FILTERS = 64
KERNEL_SIZE   = 3
FC_UNITS      = 32

# -- device selection with graceful CUDA fallback ----------------------
def get_device():
    """Attempt CUDA, fall back to CPU if GPU arch is unsupported."""
    if not torch.cuda.is_available():
        print("DEVICE: cpu (CUDA not available)")
        return torch.device("cpu")
    try:
        test = torch.zeros(1, device="cuda")
        _ = test + 1
        del test
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"DEVICE: cuda ({gpu_name})")
        return torch.device("cuda")
    except Exception as e:
        print(f"DEVICE: cpu (CUDA detected but unusable — {e})")
        print("  NOTE: Your GPU's compute capability may not be supported")
        print("  by this PyTorch build. The model will run on CPU instead.")
        return torch.device("cpu")

DEVICE = get_device()

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

TARGET_COL = "Target: Buy/hold/sell = 1/0/-1 0.3% margin"

# encode NLP_predict
df["NLP_encoded"] = (df["NLP_predict"] == "buy").astype(int)
NUMERIC_FEATURES.append("NLP_encoded")

# binary target: buy=1, else=0
df["target_binary"] = (df[TARGET_COL] == 1).astype(int)
df = df.dropna(subset=[TARGET_COL])
print(f"After dropping NaN targets: {df.shape}")

# fill missing numeric values
for col in NUMERIC_FEATURES:
    df[col] = df[col].ffill().bfill()
df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

print(f"Target distribution:\n{df['target_binary'].value_counts()}")
print(f"Features used ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")

dates = df["observation_date"].values

# =====================================================================
# 2.  SCALE
# =====================================================================
scaler = MinMaxScaler()
feature_data = scaler.fit_transform(df[NUMERIC_FEATURES].values)
target_data  = df["target_binary"].values

# =====================================================================
# 3.  SEQUENCES
# =====================================================================
def create_sequences(features, targets, seq_len):
    X, y, idx = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(targets[i])
        idx.append(i)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(idx)

X, y, seq_idx = create_sequences(feature_data, target_data, SEQ_LEN)
seq_dates = dates[seq_idx]
print(f"\nSequence dataset - X: {X.shape}, y: {y.shape}")

split = int(len(X) * (1 - TEST_RATIO))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train, dates_test = seq_dates[:split], seq_dates[split:]

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# =====================================================================
# 4.  PYTORCH DATASET & MODEL
# =====================================================================
class SP500Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SP500Dataset(X_train, y_train)
test_ds  = SP500Dataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# validation split from training data (last 15%)
val_split = int(len(X_train) * 0.85)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]
tr_loader  = DataLoader(SP500Dataset(X_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(SP500Dataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)


class CNNBinaryClassifier(nn.Module):
    """
    1D CNN for temporal sequence classification.
    Input shape: (batch, seq_len, n_features)
    Conv1D expects: (batch, channels, length) so we permute features->channels.
    """
    def __init__(self, n_features, seq_len, conv1_filters, conv2_filters,
                 conv3_filters, kernel_size, fc_units, dropout):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features, conv1_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(conv2_filters, conv3_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(conv3_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Global average pooling reduces (batch, conv3_filters, L) -> (batch, conv3_filters)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(conv3_filters, fc_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> permute to (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.global_pool(x).squeeze(-1)   # (batch, conv3_filters)
        x = self.classifier(x)
        return x.squeeze(-1)                   # raw logits

    def predict_proba(self, x):
        """Return sigmoid probabilities for inference."""
        return torch.sigmoid(self.forward(x))


model = CNNBinaryClassifier(
    n_features=X.shape[2],
    seq_len=SEQ_LEN,
    conv1_filters=CONV1_FILTERS,
    conv2_filters=CONV2_FILTERS,
    conv3_filters=CONV3_FILTERS,
    kernel_size=KERNEL_SIZE,
    fc_units=FC_UNITS,
    dropout=DROPOUT_RATE,
).to(DEVICE)

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# class weighting to handle imbalance (more weight to minority "buy" class)
n_sell = (y_train == 0).sum()
n_buy  = (y_train == 1).sum()
pos_weight = torch.tensor([n_sell / n_buy], dtype=torch.float32).to(DEVICE)
print(f"Class weight for Buy: {pos_weight.item():.2f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)

# =====================================================================
# 5.  TRAIN
# =====================================================================
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = torch.sigmoid(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
        train_correct += ((preds >= 0.5).float() == yb).sum().item()
        train_total += len(yb)

    # --- validate ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = torch.sigmoid(logits)
            val_loss += loss.item() * len(yb)
            val_correct += ((preds >= 0.5).float() == yb).sum().item()
            val_total += len(yb)

    avg_train_loss = train_loss / train_total
    avg_val_loss   = val_loss / val_total
    train_acc = train_correct / train_total
    val_acc   = val_correct / val_total

    history["loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    scheduler.step(avg_val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# restore best weights
if best_state is not None:
    model.load_state_dict(best_state)

# =====================================================================
# 6.  EVALUATE
# =====================================================================
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

model.eval()
with torch.no_grad():
    y_conf_test = model.predict_proba(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    y_conf_train = model.predict_proba(torch.tensor(X_train).to(DEVICE)).cpu().numpy()

y_pred_test = (y_conf_test >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec  = recall_score(y_test, y_pred_test, zero_division=0)
f1   = f1_score(y_test, y_pred_test, zero_division=0)
auc  = roc_auc_score(y_test, y_conf_test)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_test, y_pred_test, target_names=['Sell/Hold (0)', 'Buy (1)'])}")

# =====================================================================
# 7.  CONFIDENCE SCORE PLOTS
# =====================================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

# 7a. Confidence scores over time (test set)
fig, ax = plt.subplots(figsize=(16, 6))
test_dates_pd = pd.to_datetime(dates_test)
ax.plot(test_dates_pd, y_conf_test, linewidth=0.8, color="#2196F3",
        alpha=0.8, label="Confidence Score (P(Buy))")
ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1,
           label="Decision Threshold (0.5)")
ax.fill_between(test_dates_pd, 0.5, y_conf_test,
                where=(y_conf_test >= 0.5), alpha=0.15, color="green",
                label="Buy Region")
ax.fill_between(test_dates_pd, 0.5, y_conf_test,
                where=(y_conf_test < 0.5), alpha=0.15, color="red",
                label="Sell/Hold Region")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Confidence Score", fontsize=12)
ax.set_title("CNN Binary Classifier - Confidence Scores Over Time (Test Set)",
             fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confidence_scores_timeline.png"), dpi=200)
print("  Saved: confidence_scores_timeline.png")

# 7b. Confidence distribution histogram
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(y_conf_test[y_test == 1], bins=50, alpha=0.6, color="green",
         label="Actual Buy (1)", density=True)
ax2.hist(y_conf_test[y_test == 0], bins=50, alpha=0.6, color="red",
         label="Actual Sell/Hold (0)", density=True)
ax2.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5,
            label="Threshold")
ax2.set_xlabel("Confidence Score", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Confidence Score Distribution by True Class", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "confidence_distribution.png"), dpi=200)
print("  Saved: confidence_distribution.png")

# 7c. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_conf_test)
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.plot(fpr, tpr, color="#2196F3", linewidth=2,
         label=f"ROC Curve (AUC = {auc:.4f})")
ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
ax3.set_xlabel("False Positive Rate", fontsize=12)
ax3.set_ylabel("True Positive Rate", fontsize=12)
ax3.set_title("ROC Curve - CNN Binary Classifier", fontsize=14)
ax3.legend(loc="lower right", fontsize=12)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=200)
print("  Saved: roc_curve.png")

# 7d. Training history
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
ax4a.plot(history["loss"], label="Train Loss")
ax4a.plot(history["val_loss"], label="Val Loss")
ax4a.set_xlabel("Epoch")
ax4a.set_ylabel("Loss")
ax4a.set_title("Training & Validation Loss")
ax4a.legend()
ax4a.grid(True, alpha=0.3)

ax4b.plot(history["accuracy"], label="Train Accuracy")
ax4b.plot(history["val_accuracy"], label="Val Accuracy")
ax4b.set_xlabel("Epoch")
ax4b.set_ylabel("Accuracy")
ax4b.set_title("Training & Validation Accuracy")
ax4b.legend()
ax4b.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=200)
print("  Saved: training_history.png")

# 7e. Confidence scatter by true class
fig5, ax5 = plt.subplots(figsize=(10, 6))
colors = ["red" if yy == 0 else "green" for yy in y_test]
ax5.scatter(range(len(y_conf_test)), y_conf_test, c=colors,
            alpha=0.4, s=10, edgecolors="none")
ax5.axhline(y=0.5, color="black", linestyle="--", linewidth=1.5)
ax5.set_xlabel("Test Sample Index", fontsize=12)
ax5.set_ylabel("Confidence Score", fontsize=12)
ax5.set_title("Confidence Scores - Green=Buy(actual), Red=Sell/Hold(actual)",
              fontsize=14)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "confidence_scatter.png"), dpi=200)
print("  Saved: confidence_scatter.png")

# =====================================================================
# 8.  SAVE MODEL & RESULTS
# =====================================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "cnn_binary_model.pth"))
print(f"\n  Model saved: cnn_binary_model.pth")

results_df = pd.DataFrame({
    "date": dates_test,
    "confidence_score": y_conf_test,
    "predicted_label": y_pred_test,
    "actual_label": y_test.astype(int),
    "correct": (y_pred_test == y_test).astype(int),
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
print("  Test predictions saved: test_predictions.csv")

plt.close("all")
print("\n" + "=" * 60)
print("DONE - All outputs saved to:", OUTPUT_DIR)
print("=" * 60)
