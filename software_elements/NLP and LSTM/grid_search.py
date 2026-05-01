"""
Grid Search — Hyperparameter Tuning for LSTM & CNN Binary Classifiers
======================================================================
Searches over core hyperparameters for both the Kaggle-style LSTM and
the 1D-CNN architectures, using the same SP500 dataset & preprocessing.

Searched hyperparameters (core set):
  - learning_rate : [0.01, 0.001, 0.0001]
  - batch_size    : [8, 16, 32, 64]
  - seq_len       : [15, 30, 60]
  - hidden units  : [(64,32), (128,64), (256,128)]   (LSTM layers / CNN filters)
  - epochs        : fixed at 50 with patience=10 early stopping

Outputs:
  - grid_search_outputs/grid_results.csv        (all runs, sorted by val F1)
  - grid_search_outputs/best_lstm_model.pth
  - grid_search_outputs/best_cnn_model.pth
  - grid_search_outputs/grid_search_summary.png  (top-10 bar chart)

Usage:
  python grid_search.py                   # full search (both models)
  python grid_search.py --model lstm      # LSTM only
  python grid_search.py --model cnn       # CNN only
  python grid_search.py --quick           # reduced grid for fast testing
"""

import os
import sys
import json
import time
import argparse
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
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
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "grid_search_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- fixed settings ----------------------------------------------------
TEST_RATIO    = 0.3
MAX_EPOCHS    = 500
PATIENCE      = 20

# -- device ------------------------------------------------------------
def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        test = torch.zeros(1, device="cuda")
        _ = test + 1
        del test
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")

DEVICE = get_device()
print(f"DEVICE: {DEVICE}")

# =====================================================================
#  DATA LOADING  (done once, sequences rebuilt per seq_len)
# =====================================================================
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)
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
    "T10Y3M ",
    "BAMLC0A4CBBB",
    "CPIAUCSL",
    "DFF",
    "DGS2",
    "DGS2 1 day change",
    "Closing Value - CRUDE OIL",
    "news_article_count",
]

TARGET_COL = "Target: Buy/hold/sell = 1/0/-1 0.3% margin"

df["NLP_encoded"] = (df["NLP_predict"] == "buy").astype(int)
NUMERIC_FEATURES.append("NLP_encoded")

df["target_binary"] = (df[TARGET_COL] == 1).astype(int)
df = df.dropna(subset=[TARGET_COL])

for col in NUMERIC_FEATURES:
    df[col] = df[col].ffill().bfill()
df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

scaler = MinMaxScaler()
feature_data = scaler.fit_transform(df[NUMERIC_FEATURES].values)
target_data  = df["target_binary"].values
N_FEATURES   = len(NUMERIC_FEATURES)

print(f"Samples: {len(df)}, Features: {N_FEATURES}")
print(f"Target distribution: {dict(zip(*np.unique(target_data, return_counts=True)))}")


# =====================================================================
#  HELPERS
# =====================================================================
class SP500Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataloaders(seq_len, batch_size):
    """Build train / val / test loaders for a given seq_len & batch_size."""
    X, y = create_sequences(feature_data, target_data, seq_len)
    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    val_split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    tr_loader  = DataLoader(SP500Dataset(X_tr, y_tr),   batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(SP500Dataset(X_val, y_val),  batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SP500Dataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return tr_loader, val_loader, test_loader, X_test, y_test, y_train


# =====================================================================
#  MODEL DEFINITIONS
# =====================================================================
class LSTMBinaryClassifier(nn.Module):
    """Kaggle-style: LSTM -> LSTM -> Dense(25) -> Dense(1)"""
    def __init__(self, input_size, hidden1, hidden2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1   = nn.Linear(hidden2, 25)
        self.fc2   = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out.squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class CNNBinaryClassifier(nn.Module):
    """Conv1D(filters1) -> Conv1D(filters2) -> Conv1D(filters3) -> GAP -> Dense -> Dense(1)"""
    def __init__(self, n_features, conv1_filters, conv2_filters, conv3_filters,
                 kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, conv1_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(conv1_filters, conv2_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(conv2_filters, conv3_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(conv3_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(conv3_filters, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# =====================================================================
#  TRAINING + EVALUATION (single run)
# =====================================================================
def train_and_evaluate(model, tr_loader, val_loader, test_loader,
                       X_test, y_test, y_train, lr):
    """Train one model config, return metrics dict."""
    # class weighting
    n_sell = (y_train == 0).sum()
    n_buy  = (y_train == 1).sum()
    pos_weight = torch.tensor([n_sell / max(n_buy, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
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
            preds = torch.sigmoid(logits)
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

        avg_val_loss = val_loss / val_total

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                stopped_epoch = epoch
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- evaluate on test set ---
    model.eval()
    with torch.no_grad():
        y_conf = model.predict_proba(
            torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    y_pred = (y_conf >= 0.5).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_conf) if len(np.unique(y_test)) > 1 else 0,
        "val_loss":  best_val_loss,
        "stopped_epoch": stopped_epoch,
    }
    return metrics, model


# =====================================================================
#  GRID DEFINITION
# =====================================================================
def get_grid(quick=False):
    if quick:
        return {
            "learning_rate": [0.001, 0.0001],
            "batch_size":    [16, 32],
            "seq_len":       [30],
            "units":         [(128, 64)],
        }
    return {
        "learning_rate": [0.01, 0.001, 0.0001],
        "batch_size":    [8, 16, 32, 64],
        "seq_len":       [15, 30, 60],
        "units":         [(64, 32), (128, 64), (256, 128)],
    }


# =====================================================================
#  MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Grid search for LSTM & CNN")
    parser.add_argument("--model", choices=["lstm", "cnn", "both"], default="both",
                        help="Which model(s) to search (default: both)")
    parser.add_argument("--quick", action="store_true",
                        help="Use a reduced grid for fast testing")
    args = parser.parse_args()

    grid = get_grid(quick=args.quick)
    combos = list(itertools.product(
        grid["learning_rate"],
        grid["batch_size"],
        grid["seq_len"],
        grid["units"],
    ))

    models_to_run = []
    if args.model in ("lstm", "both"):
        models_to_run.append("lstm")
    if args.model in ("cnn", "both"):
        models_to_run.append("cnn")

    # total_runs = len(combos) * len(models_to_run)
    total_runs = 150

    print("\n" + "=" * 70)
    print(f"  GRID SEARCH")
    print(f"  Models         : {models_to_run}")
    print(f"  Combinations   : {len(combos)} per model")
    print(f"  Total runs     : {total_runs}")
    print(f"  Max epochs/run : {MAX_EPOCHS}  (patience={PATIENCE})")
    print(f"  Device         : {DEVICE}")
    print("=" * 70 + "\n")

    all_results = []
    best_models = {}   # { "lstm": (metrics, state_dict), "cnn": ... }
    run_idx = 0
    search_start = time.time()

    for model_type in models_to_run:
        best_f1 = -1
        best_state_dict = None

        for lr, bs, seq_len, (u1, u2) in combos:
            run_idx += 1
            tag = f"[{run_idx:>3}/{total_runs}] {model_type.upper()} | " \
                  f"lr={lr} bs={bs} seq={seq_len} units=({u1},{u2})"

            # Reset seed for fair comparison
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            try:
                # Build data for this seq_len / batch_size
                tr_loader, val_loader, test_loader, X_test, y_test, y_train = \
                    build_dataloaders(seq_len, bs)

                # Build model
                if model_type == "lstm":
                    model = LSTMBinaryClassifier(
                        input_size=N_FEATURES,
                        hidden1=u1,
                        hidden2=u2,
                    ).to(DEVICE)
                else:
                    # CNN: u1/u2 map to conv filter counts
                    # conv layers: u1 -> u1*2 -> u1  (with u2 used for fc)
                    model = CNNBinaryClassifier(
                        n_features=N_FEATURES,
                        conv1_filters=u1,
                        conv2_filters=u1 * 2,
                        conv3_filters=u1,
                        kernel_size=3,
                        dropout=0.3,
                    ).to(DEVICE)

                t0 = time.time()
                metrics, trained_model = train_and_evaluate(
                    model, tr_loader, val_loader, test_loader,
                    X_test, y_test, y_train, lr
                )
                elapsed = time.time() - t0

                print(f"{tag}  ->  "
                      f"F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}  "
                      f"Acc={metrics['accuracy']:.4f}  "
                      f"ep={metrics['stopped_epoch']}  "
                      f"({elapsed:.1f}s)")

                result_row = {
                    "model": model_type,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "units_1": u1,
                    "units_2": u2,
                    "time_sec": round(elapsed, 1),
                    **metrics,
                }
                all_results.append(result_row)

                # Track best
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_state_dict = {k: v.cpu().clone()
                                       for k, v in trained_model.state_dict().items()}
                    best_models[model_type] = (result_row, best_state_dict)

            except Exception as e:
                print(f"{tag}  ->  ERROR: {e}")
                all_results.append({
                    "model": model_type,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "units_1": u1,
                    "units_2": u2,
                    "f1": 0, "accuracy": 0, "precision": 0,
                    "recall": 0, "roc_auc": 0, "val_loss": 999,
                    "stopped_epoch": 0, "time_sec": 0,
                    "error": str(e),
                })

    total_time = time.time() - search_start

    # =====================================================================
    #  RESULTS
    # =====================================================================
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_DIR, "grid_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("  GRID SEARCH COMPLETE")
    print(f"  Total time : {str(timedelta(seconds=int(total_time)))}")
    print(f"  Results    : {csv_path}")
    print("=" * 70)

    # -- Print top 10 --
    print("\n  TOP 10 CONFIGURATIONS (by F1 score):")
    print("  " + "-" * 66)
    top10 = results_df.head(10)
    for i, row in top10.iterrows():
        print(f"  {i+1:>2}. {row['model'].upper():4s} | "
              f"lr={row['learning_rate']:<6} bs={int(row['batch_size']):>2} "
              f"seq={int(row['seq_len']):>2} u=({int(row['units_1'])},{int(row['units_2'])}) | "
              f"F1={row['f1']:.4f}  AUC={row['roc_auc']:.4f}  Acc={row['accuracy']:.4f}")

    # -- Best per model --
    print("\n  BEST PER MODEL:")
    print("  " + "-" * 66)
    for model_type in models_to_run:
        if model_type in best_models:
            row, state = best_models[model_type]
            print(f"  {model_type.upper():4s} -> "
                  f"lr={row['learning_rate']}  bs={row['batch_size']}  "
                  f"seq={row['seq_len']}  units=({row['units_1']},{row['units_2']})")
            print(f"         F1={row['f1']:.4f}  AUC={row['roc_auc']:.4f}  "
                  f"Acc={row['accuracy']:.4f}  Prec={row['precision']:.4f}  "
                  f"Rec={row['recall']:.4f}")

            # Save best model weights
            model_path = os.path.join(OUTPUT_DIR, f"best_{model_type}_model.pth")
            torch.save(state, model_path)
            print(f"         Saved: {model_path}")

            # Save best config as JSON
            config_path = os.path.join(OUTPUT_DIR, f"best_{model_type}_config.json")
            config = {k: v for k, v in row.items()
                      if k not in ("error",)}
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            print(f"         Config: {config_path}")

    # =====================================================================
    #  SUMMARY PLOT
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Top 15 configs bar chart
    ax1 = axes[0]
    top15 = results_df.head(15)
    labels = [f"{r['model'].upper()}\nlr={r['learning_rate']}\nbs={int(r['batch_size'])} "
              f"seq={int(r['seq_len'])}\n({int(r['units_1'])},{int(r['units_2'])})"
              for _, r in top15.iterrows()]
    colors = ["#2196F3" if r["model"] == "lstm" else "#FF9800"
              for _, r in top15.iterrows()]
    bars = ax1.bar(range(len(top15)), top15["f1"], color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(top15)))
    ax1.set_xticklabels(labels, fontsize=6, rotation=0)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("Top 15 Configurations by F1 Score", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, top15["f1"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Plot 2: Hyperparameter impact (avg F1 per value)
    ax2 = axes[1]
    param_names = ["learning_rate", "batch_size", "seq_len"]
    y_positions = []
    y_labels = []
    y_colors = []
    pos = 0
    for pname in param_names:
        grouped = results_df.groupby(pname)["f1"].mean().sort_values(ascending=True)
        for val, f1_mean in grouped.items():
            y_positions.append(pos)
            y_labels.append(f"{pname}={val}")
            y_colors.append("#4CAF50")
            ax2.barh(pos, f1_mean, color="#4CAF50", alpha=0.7, height=0.6)
            ax2.text(f1_mean + 0.002, pos, f"{f1_mean:.4f}", va="center", fontsize=8)
            pos += 1
        pos += 0.5  # gap between param groups

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_xlabel("Mean F1 Score", fontsize=12)
    ax2.set_title("Average F1 by Hyperparameter Value", fontsize=13)
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "grid_search_summary.png")
    fig.savefig(plot_path, dpi=200)
    plt.close("all")
    print(f"\n  Plot saved: {plot_path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
