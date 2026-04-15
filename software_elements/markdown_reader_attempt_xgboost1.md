# attempt_xgboost1.py -- XGBoost for S&P 500 Buy/Hold/Sell

## Overview

This script implements an **XGBoost (Extreme Gradient Boosting)** classifier to predict daily S&P 500 trading decisions: **Buy (+1)**, **Hold (0)**, or **Sell (-1)**.

XGBoost was chosen as the **"highest accuracy" method** because it consistently dominates tabular-data benchmarks -- it is the most-used algorithm in winning Kaggle solutions for structured/tabular data, handles class imbalance natively, and includes built-in L1/L2 regularization.

## What Was Done

### Data Source
- **File:** `SP500V3.xlsx` -> Sheet `SP500`
- **Rows:** ~2,513 daily observations
- **Target column:** `Target: Buy/hold/sell = 1/0/-1 0.3% margin`
- **Features:** Same 13 macro-financial indicators as the TensorFlow attempt

### Model Configuration
```
XGBClassifier(
    n_estimators   = 500,
    max_depth      = 5,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    gamma          = 1,
    reg_alpha      = 0.1,   (L1 regularization)
    reg_lambda     = 1.0,   (L2 regularization)
    min_child_weight = 3,
    objective      = "multi:softprob",
    early_stopping_rounds = 30
)
```

### Class Imbalance Handling
- Computed inverse-frequency sample weights for each class
- Applied during training to give minority class (Sell) more influence

### Training Configuration
- **Train/Test split:** 80/20 chronological (identical to TF script)
- **Evaluation set:** Last 15% of training data (for early stopping)
- **Feature scaling:** StandardScaler (for fair comparison with TF)

### Metrics Reported (matching `project.py`)
- Accuracy
- Balanced Accuracy
- Macro F1 Score
- Per-class Precision, Recall, F1 (via classification report)
- Mean and Median prediction confidence
- Training and test label distributions

### Graphs Generated

| File | Description |
|------|-------------|
| `XGB_Confusion_Matrix.png` | Normalised confusion matrix heatmap |
| `XGB_Feature_Importance.png` | Gain-based feature importance (sorted) |
| `XGB_Confidence_Distribution.png` | Histogram of prediction confidence per class |
| `XGB_Tree_Visualization.png` | Visual plot of the first boosted tree (similar to project.py tree plot) |

## How to Run
```bash
cd software_elements
python attempt_xgboost1.py
```

## Why XGBoost for Highest Accuracy?

1. **Tabular data champion:** Academic benchmarks and competitions (Kaggle) consistently show gradient-boosted trees outperform neural networks on structured/tabular data with fewer than ~10k rows.
2. **Built-in regularization:** L1/L2 penalties and tree-depth limits prevent overfitting without needing dropout or batch normalization.
3. **Missing value handling:** XGBoost learns optimal default directions for missing values during training.
4. **Efficiency:** Orders of magnitude faster to train than deep learning on this dataset size.
5. **Interpretability:** Native feature importance scores and individual tree visualization.

## Key Differences from project.py
| Aspect | project.py | attempt_xgboost1.py |
|--------|-----------|---------------------|
| Model | Single Decision Tree | 500 Boosted Trees (ensemble) |
| Data | SP500_LSTM_data.xlsx (day_1..day_30) | SP500V3.xlsx (macro indicators) |
| Regularization | max_depth=4 only | L1, L2, gamma, min_child_weight |
| Class Balance | None | Inverse-frequency sample weights |
| Confidence | Not computed | Softmax probabilities |
