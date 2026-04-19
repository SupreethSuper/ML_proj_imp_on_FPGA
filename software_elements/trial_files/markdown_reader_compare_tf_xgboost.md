# compare_tf_xgboost.py -- TensorFlow vs XGBoost Comparison

## Overview

This script **trains both models on identical data** and produces a comprehensive head-to-head comparison of the TensorFlow DNN and XGBoost classifiers for the S&P 500 Buy/Hold/Sell prediction task.

## What Was Done

### Shared Experimental Setup
- **Dataset:** `SP500V3.xlsx` -> Sheet `SP500` (~2,513 rows, 13 features)
- **Train/Test split:** 80/20 chronological (same split index for both)
- **Feature scaling:** StandardScaler (same scaler instance for both)
- **Target:** Buy (+1) / Hold (0) / Sell (-1) with 0.3% margin

### Models Trained
1. **TensorFlow DNN** -- 4-layer feed-forward network (128->64->32->3) with BatchNorm, Dropout, Adam optimizer, early stopping
2. **XGBoost** -- 500 boosted trees, depth 5, with L1/L2 regularization, class-weight balancing, early stopping

### Metrics Compared
| Metric | What It Measures |
|--------|-----------------|
| Accuracy | Overall correct predictions / total |
| Balanced Accuracy | Average per-class recall (handles imbalance) |
| Macro F1 | Harmonic mean of precision & recall, macro-averaged |
| Macro Precision | Average precision across all classes |
| Macro Recall | Average recall across all classes |
| Mean Confidence | Average of the winning class's softmax probability |
| Median Confidence | Median of the winning class's softmax probability |

### Graphs Generated

| File | Description |
|------|-------------|
| `CMP_Metrics_BarChart.png` | Grouped bar chart comparing Accuracy, Balanced Accuracy, F1, Precision, Recall |
| `CMP_Confusion_Matrices.png` | Side-by-side normalized confusion matrices |
| `CMP_Confidence_BoxPlot.png` | Box plot of confidence distributions for both models |
| `CMP_Feature_Importance_Compared.png` | Overlaid feature importance (TF permutation vs XGB gain, normalized) |
| `CMP_Per_Class_F1.png` | Per-class F1 score comparison (Sell / Hold / Buy) |

## How to Run
```bash
cd software_elements
python compare_tf_xgboost.py
```

## Expected Output
The script prints a formatted table like:
```
Metric                     TensorFlow      XGBoost      Winner
------------------------------------------------------------
Accuracy                       0.XXXX       0.XXXX        ...
Balanced Accuracy              0.XXXX       0.XXXX        ...
Macro F1                       0.XXXX       0.XXXX        ...
...
```

It also prints the full classification report for each model and a **Final Verdict** indicating which model wins on the majority of key metrics.

## Interpretation Guide

- **Higher Accuracy** = more overall correct predictions
- **Higher Balanced Accuracy** = better at handling all classes equally (important since Sell class is under-represented)
- **Higher Macro F1** = better balance of precision and recall across all classes
- **Higher Confidence** = model is more "sure" of its predictions (but overconfidence can be misleading)
- **Feature Importance** differences reveal what each model "pays attention to" -- if they agree on top features, the signal is robust

## Summary of Approach
Both models are trained and evaluated under identical conditions to ensure a fair comparison. The comparison script does NOT reuse saved models -- it retrains from scratch to guarantee reproducibility and identical data splits.
