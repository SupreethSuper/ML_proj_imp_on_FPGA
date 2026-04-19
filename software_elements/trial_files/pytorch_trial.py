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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 0. Command-line Arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="PyTorch MLP classifier for S&P 500 with configurable date window"
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
    default=200,
    help="Number of training epochs. Default: 200"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size. Default: 32"
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Learning rate. Default: 1e-3"
)
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Early stopping patience. Default: 20"
)
args = parser.parse_args()

# ============================================================
# 1. Load & Prepare Data
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "SP500V3.xlsx")
SHEET = "SP500"

print(f"Reading sheet '{SHEET}' from '{DATA_PATH}' ...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")

date_col = "observation_date"
df[date_col] = pd.to_datetime(df[date_col])

start_date = pd.to_datetime(args.start_date)
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

exclude = {
    "observation_date",
    "Target: Raw next-day return",
    "Target: Up or Down - 1 next day UP, 0 next day down/flat",
    TARGET_COL
}
feature_cols = [c for c in df.columns if c not in exclude]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feature_cols)

X = df[feature_cols].values.astype(np.float32)
Y = df[TARGET_COL].astype(int).values   # -1, 0, 1
Y_shifted = (Y + 1).astype(np.int64)    # 0, 1, 2

CLASS_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]

print(f"Final dataset: {len(X)} rows | {len(feature_cols)} features")
print(f"Epochs: {args.epochs}\n")

# ============================================================
# 2. Chronological Train / Val / Test Split
# ============================================================
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
y_train = Y_shifted[:train_end]

X_val = X[train_end:val_end]
y_val = Y_shifted[train_end:val_end]

X_test = X[val_end:]
y_test = Y_shifted[val_end:]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scaling for neural nets
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_val_sc = scaler.transform(X_val).astype(np.float32)
X_test_sc = scaler.transform(X_test).astype(np.float32)

# ============================================================
# 3. Dataset / Loader
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TabularDataset(X_train_sc, y_train)
val_ds = TabularDataset(X_val_sc, y_val)
test_ds = TabularDataset(X_test_sc, y_test)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# ============================================================
# 4. Model
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MLPClassifier(input_dim=X_train_sc.shape[1], num_classes=3).to(device)

# Class weights for imbalance
counter = Counter(y_train)
total = len(y_train)
class_weights = [total / (len(counter) * counter[c]) for c in range(3)]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# ============================================================
# 5. Training utilities
# ============================================================
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * xb.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    return avg_loss, all_targets, all_preds, all_probs

# ============================================================
# 6. Train with early stopping
# ============================================================
best_val_loss = float("inf")
best_state = None
best_epoch = -1
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

for epoch in range(1, args.epochs + 1):
    train_loss, y_train_ep, y_pred_train_ep, _ = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, y_val_ep, y_pred_val_ep, _ = run_epoch(
        model, val_loader, criterion
    )

    scheduler.step(val_loss)

    train_acc = accuracy_score(y_train_ep, y_pred_train_ep)
    val_acc = accuracy_score(y_val_ep, y_pred_val_ep)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:03d}/{args.epochs} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

print(f"\nTraining completed for all {args.epochs} epochs.")

# ============================================================
# 7. Test Evaluation
# ============================================================
test_loss, y_test_eval, y_pred_eval, y_proba = run_epoch(model, test_loader, criterion)
confidence = np.max(y_proba, axis=1)

y_test_orig = y_test_eval - 1
y_pred_orig = y_pred_eval - 1

acc = accuracy_score(y_test_orig, y_pred_orig)
bal_acc = balanced_accuracy_score(y_test_orig, y_pred_orig)
f1_mac = f1_score(y_test_orig, y_pred_orig, average="macro")

print("\n" + "=" * 50)
print(f"--- PyTorch MLP Evaluation ({days_count}-day window) ---")
print("=" * 50)
print(f"Accuracy          : {acc:.4f}")
print(f"Balanced Accuracy : {bal_acc:.4f}")
print(f"Macro F1          : {f1_mac:.4f}")
print(f"Test Loss         : {test_loss:.4f}")
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
# 8. Visualization 1 – Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test_orig, y_pred_orig, labels=[-1, 0, 1])

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
    title=f"PyTorch MLP ({days_count}-day) – Normalized Confusion Matrix"
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "MLP_Confusion_Matrix.png"), dpi=300)
print("\nSaved MLP_Confusion_Matrix.png")
plt.close(fig)

# ============================================================
# 9. Visualization 2 – Confidence Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = y_pred_eval == cls_idx
    if mask.sum() > 0:
        ax.hist(confidence[mask], bins=30, alpha=0.5, label=cls_name)

ax.set_title(f"PyTorch MLP ({days_count}-day) – Prediction Confidence Distribution")
ax.set_xlabel("Confidence (max softmax probability)")
ax.set_ylabel("Count")
ax.legend()

fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "MLP_Confidence_Distribution.png"), dpi=300)
print("Saved MLP_Confidence_Distribution.png")
plt.close(fig)

# ============================================================
# 10. Visualization 3 – Training Curves
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history["train_loss"], label="Train Loss")
ax.plot(history["val_loss"], label="Val Loss")
ax.set_title("MLP Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "MLP_Training_Loss.png"), dpi=300)
print("Saved MLP_Training_Loss.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history["train_acc"], label="Train Accuracy")
ax.plot(history["val_acc"], label="Val Accuracy")
ax.set_title("MLP Training Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "MLP_Training_Accuracy.png"), dpi=300)
print("Saved MLP_Training_Accuracy.png")
plt.close(fig)

print("\n✓ PyTorch MLP script completed successfully.")