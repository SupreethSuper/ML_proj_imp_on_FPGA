# attempt_tensorflow1.py -- TensorFlow DNN for S&P 500 Buy/Hold/Sell

## Overview

This script implements a **TensorFlow / Keras Deep Neural Network (DNN)** to classify daily S&P 500 observations into three trading decisions: **Buy (+1)**, **Hold (0)**, or **Sell (-1)**, using data from the `SP500V3.xlsx` spreadsheet (sheet: `SP500`).

## What Was Done

### Data Source
- **File:** `SP500V3.xlsx` -> Sheet `SP500`
- **Rows:** ~2,513 daily observations (April 2016 onward)
- **Target column:** `Target: Buy/hold/sell = 1/0/-1 0.3% margin`

### Features Used (13 columns)
| Feature | Description |
|---------|-------------|
| SP500 | Daily closing index level |
| SP500 1 day return | Percentage change from prior day |
| SP500 3 day momentum | 3-day rolling return |
| VIX | CBOE Volatility Index |
| VIX 1 day change | Daily VIX delta |
| DGS10 | 10-Year Treasury Yield |
| DGS10 1 day change | Daily yield delta |
| T10Y3M | 10Y-3M Treasury Spread |
| BAMLC0A4CBBB | BBB Corporate Bond Spread |
| CPIAUCSL | Consumer Price Index |
| DFF | Federal Funds Rate |
| DGS2 | 2-Year Treasury Yield |
| DGS2 1 day change | Daily 2Y yield delta |

### Model Architecture
```
Input (13 features)
  -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.3)
  -> Dense(64, ReLU)  -> BatchNorm -> Dropout(0.3)
  -> Dense(32, ReLU)  -> Dropout(0.2)
  -> Dense(3, Softmax)   [Buy / Hold / Sell]
```

### Training Configuration
- **Train/Test split:** 80/20 chronological (no shuffle, prevents time-series data leakage)
- **Validation:** 15% of training set
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Early stopping:** patience=15, restore best weights
- **Learning rate reduction:** factor=0.5, patience=5
- **Max epochs:** 150, batch size 64
- **Feature scaling:** StandardScaler (zero mean, unit variance)

### Metrics Reported (matching `project.py`)
- Accuracy
- Balanced Accuracy
- Macro F1 Score
- Per-class Precision, Recall, F1 (via classification report)
- Mean and Median prediction confidence (softmax probability)
- Training and test label distributions

### Graphs Generated

| File | Description |
|------|-------------|
| `TF_Training_History.png` | Loss and accuracy curves over training epochs |
| `TF_Confusion_Matrix.png` | Normalised confusion matrix heatmap |
| `TF_Feature_Importance.png` | Permutation-based feature importance bar chart |
| `TF_Confidence_Distribution.png` | Histogram of prediction confidence per class |

## How to Run
```bash
cd software_elements
python attempt_tensorflow1.py
```

## Key Differences from project.py
| Aspect | project.py | attempt_tensorflow1.py |
|--------|-----------|----------------------|
| Model | Decision Tree (sklearn) | Deep Neural Network (TensorFlow/Keras) |
| Data | SP500_LSTM_data.xlsx (day_1..day_30) | SP500V3.xlsx (macro indicators) |
| Scaling | None required | StandardScaler |
| Confidence | Not computed | Softmax probabilities |
| Feature Importance | Tree-based (native) | Permutation-based |
