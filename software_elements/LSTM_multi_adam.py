"""
LSTM Multi-Class Classification for SP500 Buy/Hold/Sell Prediction
===================================================================
Dataset : SP500V3.xlsx  (sheet "SP500")
Target  : Buy/hold/sell column (Buy=1, Hold=0, Sell=-1) -> 3 classes
Features: All available numeric indicators + NLP_predict + news_article_count
Sequence: 30-day lookback window
Optimizer: Adam with adaptive ReduceLROnPlateau scheduler
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
import torch.nn.functional as F
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
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets", "SP500V3.xlsx")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "lstm_multi_adam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- hyperparameters ---------------------------------------------------
SEQ_LEN       = 30
TEST_RATIO    = 0.3
EPOCHS        = 100
BATCH_SIZE    = 50
LSTM_UNITS_1  = 260
# LSTM_UNITS_1  = 128
LSTM_UNITS_2  = 128
# LSTM_UNITS_2  = 64
DROPOUT_RATE  = 0.3
LEARNING_RATE = 0.001
PATIENCE      = 30
NUM_CLASSES   = 3              # Buy, Hold, Sell

# class label mapping:  original -> index
#   -1 (Sell) -> 0
#    0 (Hold) -> 1
#    1 (Buy)  -> 2
CLASS_NAMES = ["Sell", "Hold", "Buy"]
CLASS_COLORS = ["#E53935", "#FFA726", "#43A047"]  # red, orange, green

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
        print(f"DEVICE: cpu (CUDA detected but unusable - {e})")
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

# encode NLP_predict (buy=1, sell=0)
df["NLP_encoded"] = (df["NLP_predict"] == "buy").astype(int)
NUMERIC_FEATURES.append("NLP_encoded")

# multi-class target: map {-1, 0, 1} -> {0, 1, 2}
#   -1 (Sell) -> 0
#    0 (Hold) -> 1
#    1 (Buy)  -> 2
df = df.dropna(subset=[TARGET_COL])
df["target_multi"] = df[TARGET_COL].map({-1: 0, 0: 1, 1: 2}).astype(int)
print(f"After dropping NaN targets: {df.shape}")

# fill missing numeric values
for col in NUMERIC_FEATURES:
    df[col] = df[col].ffill().bfill()
df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

print(f"Target distribution:")
for idx, name in enumerate(CLASS_NAMES):
    count = (df["target_multi"] == idx).sum()
    print(f"  {name} ({idx}): {count}")
print(f"Features used ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")

dates = df["observation_date"].values

# =====================================================================
# 2.  SCALE
# =====================================================================
scaler = MinMaxScaler()
feature_data = scaler.fit_transform(df[NUMERIC_FEATURES].values)
target_data  = df["target_multi"].values

# =====================================================================
# 3.  SEQUENCES
# =====================================================================
def create_sequences(features, targets, seq_len):
    X, y, idx = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(targets[i])
        idx.append(i)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(idx)

X, y, seq_idx = create_sequences(feature_data, target_data, SEQ_LEN)
seq_dates = dates[seq_idx]
print(f"\nSequence dataset - X: {X.shape}, y: {y.shape}")

split = int(len(X) * (1 - TEST_RATIO))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train, dates_test = seq_dates[:split], seq_dates[split:]

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train dist: {np.bincount(y_train, minlength=NUM_CLASSES)}")
print(f"Test  dist: {np.bincount(y_test, minlength=NUM_CLASSES)}")

# =====================================================================
# 4.  PYTORCH DATASET & MODEL
# =====================================================================
class SP500Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
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


class LSTMMultiClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, dropout, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True,
                             bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 32)
        self.relu  = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        self.fc2   = nn.Linear(32, num_classes)   # 3-class output

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])   # last time step
        out = self.relu(self.fc1(out))
        out = self.drop3(out)
        out = self.fc2(out)               # raw logits (3 classes)
        return out

    def predict_proba(self, x):
        """Return softmax probabilities for inference."""
        return F.softmax(self.forward(x), dim=-1)


model = LSTMMultiClassifier(
    input_size=X.shape[2],
    hidden1=LSTM_UNITS_1,
    hidden2=LSTM_UNITS_2,
    dropout=DROPOUT_RATE,
    num_classes=NUM_CLASSES,
).to(DEVICE)

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# class weighting to handle imbalance
class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: Sell={class_weights[0]:.2f}, Hold={class_weights[1]:.2f}, Buy={class_weights[2]:.2f}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Adam optimizer with adaptive ReduceLROnPlateau scheduler
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

history = {
    "loss": [], "val_loss": [],
    "accuracy": [], "val_accuracy": [],
    "test_accuracy": [],
    "lr": [],
}
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
        preds = logits.argmax(dim=-1)
        train_correct += (preds == yb).sum().item()
        train_total += len(yb)

    # --- validate ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    test_correct_epoch, test_total_epoch = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * len(yb)
            preds = logits.argmax(dim=-1)
            val_correct += (preds == yb).sum().item()
            val_total += len(yb)

        # --- test accuracy per epoch (for Train vs Test plot) ---
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=-1)
            test_correct_epoch += (preds == yb).sum().item()
            test_total_epoch += len(yb)

    avg_train_loss = train_loss / train_total
    avg_val_loss   = val_loss / val_total
    train_acc = train_correct / train_total
    val_acc   = val_correct / val_total
    test_acc  = test_correct_epoch / test_total_epoch

    history["loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)
    history["test_accuracy"].append(test_acc)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    history["lr"].append(current_lr)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | LR: {current_lr:.2e}")

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
    # softmax probabilities: shape (N, 3)
    conf_test  = model.predict_proba(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    conf_train = model.predict_proba(torch.tensor(X_train).to(DEVICE)).cpu().numpy()

y_pred_test  = conf_test.argmax(axis=1)
y_pred_train = conf_train.argmax(axis=1)

acc  = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

print(f"Accuracy          : {acc:.4f}")
print(f"Precision (wt avg): {prec:.4f}")
print(f"Recall    (wt avg): {rec:.4f}")
print(f"F1 Score  (wt avg): {f1:.4f}")

# one-vs-rest AUC
try:
    auc = roc_auc_score(y_test, conf_test, multi_class="ovr", average="weighted")
    print(f"ROC-AUC   (wt avg): {auc:.4f}")
except Exception:
    auc = None
    print("ROC-AUC   : could not compute (class missing in test set)")

print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_test, y_pred_test, target_names=CLASS_NAMES)}")

final_train_acc = accuracy_score(y_train, y_pred_train)
final_test_acc  = acc
print(f"Final Train Accuracy: {final_train_acc:.4f}")
print(f"Final Test  Accuracy: {final_test_acc:.4f}")

# =====================================================================
# 7.  CONFIDENCE SCORE PLOTS
# =====================================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

PREFIX = "LSTM_multi_"
SUFFIX = "_adam.png"

# ---- 7a. Confidence scores over time (test set) - all 3 classes ----
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
test_dates_pd = pd.to_datetime(dates_test)

for cls_idx, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, CLASS_COLORS)):
    ax.plot(test_dates_pd, conf_test[:, cls_idx], linewidth=0.8,
            color=color, alpha=0.8, label=f"P({name})")
    ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Random baseline (0.33)")
    ax.set_ylabel(f"P({name})", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[0].set_title("LSTM Multi-Class (Adam) - Confidence Scores Over Time (Test Set)",
                   fontsize=14)
axes[-1].set_xlabel("Date", fontsize=12)
fig.tight_layout()
fname = f"{PREFIX}confidence_timeline{SUFFIX}"
fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7b. Stacked confidence area chart ----
fig2, ax2 = plt.subplots(figsize=(16, 6))
ax2.stackplot(test_dates_pd,
              conf_test[:, 0], conf_test[:, 1], conf_test[:, 2],
              labels=CLASS_NAMES, colors=CLASS_COLORS, alpha=0.7)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Confidence Score", fontsize=12)
ax2.set_title("LSTM Multi-Class (Adam) - Stacked Confidence Scores (Test Set)", fontsize=14)
ax2.legend(loc="upper right")
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fname = f"{PREFIX}confidence_stacked{SUFFIX}"
fig2.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7c. Confidence distribution per class ----
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
for cls_idx, (ax, name, color) in enumerate(zip(axes3, CLASS_NAMES, CLASS_COLORS)):
    for true_cls, true_name in enumerate(CLASS_NAMES):
        mask = y_test == true_cls
        ax.hist(conf_test[mask, cls_idx], bins=40, alpha=0.5,
                color=CLASS_COLORS[true_cls], label=f"Actual {true_name}",
                density=True)
    ax.set_xlabel(f"P({name})", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Distribution of P({name})", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig3.suptitle("Confidence Score Distributions by True Class (Adam)", fontsize=14, y=1.02)
fig3.tight_layout()
fname = f"{PREFIX}confidence_distribution{SUFFIX}"
fig3.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches="tight")
print(f"  Saved: {fname}")

# ---- 7d. ROC Curves (one-vs-rest) ----
fig4, ax4 = plt.subplots(figsize=(8, 8))
for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    y_binary = (y_test == cls_idx).astype(int)
    if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
        fpr_c, tpr_c, _ = roc_curve(y_binary, conf_test[:, cls_idx])
        auc_c = roc_auc_score(y_binary, conf_test[:, cls_idx])
        ax4.plot(fpr_c, tpr_c, color=color, linewidth=2,
                 label=f"{name} (AUC = {auc_c:.4f})")
ax4.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
ax4.set_xlabel("False Positive Rate", fontsize=12)
ax4.set_ylabel("True Positive Rate", fontsize=12)
ax4.set_title("ROC Curves (One-vs-Rest) - LSTM Multi-Class (Adam)", fontsize=14)
ax4.legend(loc="lower right", fontsize=11)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fname = f"{PREFIX}roc_curve{SUFFIX}"
fig4.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7e. Training history (loss) ----
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(history["loss"], label="Train Loss", linewidth=1.5)
ax5.plot(history["val_loss"], label="Val Loss", linewidth=1.5)
ax5.set_xlabel("Epoch", fontsize=12)
ax5.set_ylabel("Loss", fontsize=12)
ax5.set_title("Training & Validation Loss (Adam)", fontsize=14)
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fname = f"{PREFIX}training_loss{SUFFIX}"
fig5.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7f. Train Accuracy vs Test Accuracy ----
fig6, ax6 = plt.subplots(figsize=(10, 6))
epochs_range = range(1, len(history["accuracy"]) + 1)
ax6.plot(epochs_range, history["accuracy"], label="Train Accuracy",
         linewidth=1.5, color="#2196F3")
ax6.plot(epochs_range, history["test_accuracy"], label="Test Accuracy",
         linewidth=1.5, color="#E53935")
ax6.plot(epochs_range, history["val_accuracy"], label="Val Accuracy",
         linewidth=1.5, color="#FFA726", linestyle="--", alpha=0.7)
ax6.set_xlabel("Epoch", fontsize=12)
ax6.set_ylabel("Accuracy", fontsize=12)
ax6.set_title("Train Accuracy vs Test Accuracy (Adam)", fontsize=14)
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)
fig6.tight_layout()
fname = f"{PREFIX}train_vs_test_accuracy{SUFFIX}"
fig6.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7g. Confusion matrix heatmap ----
fig7, ax7 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_test)
im = ax7.imshow(cm, interpolation="nearest", cmap="Blues")
fig7.colorbar(im, ax=ax7)
ax7.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax7.set_xlabel("Predicted", fontsize=12)
ax7.set_ylabel("Actual", fontsize=12)
ax7.set_title("Confusion Matrix - LSTM Multi-Class (Adam)", fontsize=14)
# annotate cells
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax7.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color=color, fontsize=14, fontweight="bold")
fig7.tight_layout()
fname = f"{PREFIX}confusion_matrix{SUFFIX}"
fig7.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7h. Confidence scatter by true class ----
fig8, ax8 = plt.subplots(figsize=(12, 6))
for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    mask = y_test == cls_idx
    # plot max confidence for each sample, colored by true class
    ax8.scatter(np.where(mask)[0], conf_test[mask].max(axis=1),
                c=color, alpha=0.4, s=12, edgecolors="none", label=f"Actual {name}")
ax8.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax8.set_xlabel("Test Sample Index", fontsize=12)
ax8.set_ylabel("Max Confidence Score", fontsize=12)
ax8.set_title("Max Confidence per Sample (Adam, colored by true class)", fontsize=14)
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)
fig8.tight_layout()
fname = f"{PREFIX}confidence_scatter{SUFFIX}"
fig8.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# ---- 7i. Learning rate schedule ----
fig9, ax9 = plt.subplots(figsize=(10, 5))
ax9.plot(range(1, len(history["lr"]) + 1), history["lr"],
         linewidth=1.5, color="#9C27B0")
ax9.set_xlabel("Epoch", fontsize=12)
ax9.set_ylabel("Learning Rate", fontsize=12)
ax9.set_title("Adam Adaptive Learning Rate Schedule", fontsize=14)
ax9.set_yscale("log")
ax9.grid(True, alpha=0.3)
fig9.tight_layout()
fname = f"{PREFIX}learning_rate_schedule{SUFFIX}"
fig9.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
print(f"  Saved: {fname}")

# =====================================================================
# 8.  SAVE MODEL & RESULTS
# =====================================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_multi_adam_model.pth"))
print(f"\n  Model saved: lstm_multi_adam_model.pth")

results_df = pd.DataFrame({
    "date": dates_test,
    "conf_sell": conf_test[:, 0],
    "conf_hold": conf_test[:, 1],
    "conf_buy":  conf_test[:, 2],
    "predicted_label": y_pred_test,
    "predicted_class": [CLASS_NAMES[p] for p in y_pred_test],
    "actual_label": y_test,
    "actual_class": [CLASS_NAMES[a] for a in y_test],
    "correct": (y_pred_test == y_test).astype(int),
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
print("  Test predictions saved: test_predictions.csv")

plt.close("all")
print("\n" + "=" * 60)
print("DONE - All outputs saved to:", OUTPUT_DIR)
print("=" * 60)
