"""
LSTM Binary Classification for SP500 Buy/Sell Prediction
=========================================================
Dataset : SP500V3 - Copy.xlsx  (sheet "SP500")
Target  : Buy/hold/sell column -> binary (Buy=1, else=0)
Features: All numeric indicators + NLP_predict + news_article_count
Sequence: 30-day lookback window
Output  : Confidence-score plots via matplotlib
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
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets", "SP500V3 - Copy.xlsx")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "lstm_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- hyperparameters (grid search best) --------------------------------
SEQ_LEN       = 60
TEST_RATIO    = 0.3
EPOCHS        = 500
BATCH_SIZE    = 8
LSTM_UNITS_1  = 256
LSTM_UNITS_2  = 128
LEARNING_RATE = 0.001
PATIENCE      = 20

# -- device selection with graceful CUDA fallback ----------------------
def get_device():
    """Attempt CUDA, fall back to CPU if GPU arch is unsupported."""
    if not torch.cuda.is_available():
        print("DEVICE: cpu (CUDA not available)")
        return torch.device("cpu")
    try:
        # Run a small test tensor operation on GPU to verify kernel support
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
        print("  Check https://pytorch.org/get-started/locally/ for a")
        print("  compatible nightly build with CUDA 12.8+ / sm_120 support.")
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


class LSTMBinaryClassifier(nn.Module):
    """
    Kaggle-style sequential LSTM for binary classification.
    Architecture: LSTM(128) -> LSTM(64) -> Dense(25) -> Dense(1)
    Mirrors the Keras Sequential model from the Kaggle S&P 500 notebook,
    adapted for PyTorch and binary classification with logit output.
    """
    def __init__(self, input_size, hidden1, hidden2, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1   = nn.Linear(hidden2, 25)
        self.fc2   = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)           # all timesteps -> next LSTM
        out, _ = self.lstm2(out)
        out = out[:, -1, :]              # last time step only
        out = self.fc1(out)
        out = self.fc2(out)              # raw logits
        return out.squeeze(-1)

    def predict_proba(self, x):
        """Return sigmoid probabilities for inference."""
        return torch.sigmoid(self.forward(x))


model = LSTMBinaryClassifier(
    input_size=X.shape[2],
    hidden1=LSTM_UNITS_1,
    hidden2=LSTM_UNITS_2,
    dropout=0,
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

# -- Kaggle-style plain Adam optimizer ---------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
ax.set_title("LSTM Binary Classifier - Confidence Scores Over Time (Test Set)",
             fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confidence_scores_timeline_adam.png"), dpi=200)
print("  Saved: confidence_scores_timeline_adam.png")

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
fig2.savefig(os.path.join(OUTPUT_DIR, "confidence_distribution_adam.png"), dpi=200)
print("  Saved: confidence_distribution_adam.png")

# 7c. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_conf_test)
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.plot(fpr, tpr, color="#2196F3", linewidth=2,
         label=f"ROC Curve (AUC = {auc:.4f})")
ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
ax3.set_xlabel("False Positive Rate", fontsize=12)
ax3.set_ylabel("True Positive Rate", fontsize=12)
ax3.set_title("ROC Curve - LSTM Binary Classifier", fontsize=14)
ax3.legend(loc="lower right", fontsize=12)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "roc_curve_adam.png"), dpi=200)
print("  Saved: roc_curve_adam.png")

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
fig4.savefig(os.path.join(OUTPUT_DIR, "training_history_adam.png"), dpi=200)
print("  Saved: training_history_adam.png")

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
fig5.savefig(os.path.join(OUTPUT_DIR, "confidence_scatter_adam.png"), dpi=200)
print("  Saved: confidence_scatter_adam.png")

# =====================================================================
# 8.  SAVE MODEL & RESULTS
# =====================================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_binary_model.pth"))
print(f"\n  Model saved: lstm_binary_model.pth")

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