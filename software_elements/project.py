import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Read data from the Excel sheet
# ==========================================
file_name = 'SP500_LSTM_data.xlsx'
sheet = 'LSTM_Chunked_SP500'

print(f"Reading sheet '{sheet}' from file '{file_name}'...")
# Note: skiprows=7 is very important. It tells pandas to skip the first 7 rows
# of instructions and read the actual header starting from row 8.
df = pd.read_excel(file_name, sheet_name=sheet, skiprows=7)

# Basic cleaning: drop rows where the target column is missing
df = df.dropna(subset=['target_buy_hold_sell'])

# ==========================================
# 2. Extract features (X) and labels (Y)
# ==========================================
# Automatically select all columns starting from day_1 to day_30 as input features
feature_cols = [f'day_{i}' for i in range(1, 31)]
X = df[feature_cols]

# Extract the target column for Buy (1) / Hold (0) / Sell (-1)
Y = df['target_buy_hold_sell'].astype(int)

# ==========================================
# 3. Split the dataset in chronological order (80% train, 20% test)
# Do not shuffle, to avoid data leakage in time series
# ==========================================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
y_train = Y.iloc[:split_idx]

X_test = X.iloc[split_idx:]
y_test = Y.iloc[split_idx:]

print(f"Data split completed -> Training set size: {len(X_train)} rows, Test set size: {len(X_test)} rows")

# ==========================================
# 4. Train the decision tree model
# As discussed, decision trees do not require feature normalization
# ==========================================
# Set random_state for reproducibility, and limit max_depth to reduce overfitting
dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_model.fit(X_train, y_train)

# ==========================================
# 5. Model prediction and evaluation
# ==========================================
y_pred = dt_model.predict(X_test)

print("\n" + "="*30)
print("--- Decision Tree Model Evaluation ---")
print("="*30)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# 6. Visualization: plot the decision tree
# ==========================================
plt.figure(figsize=(20, 10))  # Set figure size

# Plot the tree
plot_tree(
    dt_model,
    filled=True,
    feature_names=X.columns,
    class_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'],
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree for S&P 500 Classification (Buy/Hold/Sell)", fontsize=16)
plt.tight_layout()

# Save the figure locally
plt.savefig("Decision_Tree_SP500.png", dpi=300)
print("\nDecision tree image saved as 'Decision_Tree_SP500.png'")

# Display the figure
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# Additional Visualization 1: confusion matrix
# ==========================================
# Explicitly set label order: -1 (Sell), 0 (Hold), 1 (Buy)
# This ensures consistency with the interpretation above
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# Final Visualization: normalized confusion matrix
# ==========================================


# ==========================================
# Additional Visualization 2: feature importance
# ==========================================
plt.figure(figsize=(14, 6))
importances = dt_model.feature_importances_

# Plot the importance score of each daily feature
plt.bar(X.columns, importances, color='teal')
plt.title("Feature Importance - Which days drive the decision?", fontsize=16)
plt.xlabel("Features (Day 1 to Day 30)", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to avoid overlap
plt.tight_layout()

plt.savefig("Feature_Importance_DT.png", dpi=300)
print("Feature importance plot saved as 'Feature_Importance_DT.png'")
plt.show()

print("Training label distribution:")
print(y_train.value_counts().sort_index())

print("\nTest label distribution:")
print(y_test.value_counts().sort_index())

from sklearn.metrics import balanced_accuracy_score, f1_score

print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
print("Macro F1:", round(f1_score(y_test, y_pred, average='macro'), 4))
