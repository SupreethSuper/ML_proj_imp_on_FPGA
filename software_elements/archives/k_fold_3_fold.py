import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Use the sibling file reader module
from file_reader_csv import SP500Reader

class K_fold_sp500:
    """
    Perform K-fold classification on the S&P500 index data.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = SP500Reader(filepath)
        self.X = None
        self.y = None

    def prepare_data(self):
        """
        Load the CSV and create binary classification target and features.
        Target: 1 if the index goes up the next day, 0 if it goes down.
        Features: Lagged values of the S&P500 index.
        """
        df = self.reader.get_data()

        # Create target variable: Did the market go up the next day?
        # shift(-1) moves tomorrow's price to today's row
        df['Next_SP500'] = df['S&P500'].shift(-1)
        df['Target'] = (df['Next_SP500'] > df['S&P500']).astype(int)

        # Create features: Lag 0 through 4
        # Note: 'S&P500' is Lag 0 (today's price)
        df['Lag_1'] = df['S&P500'].shift(1)
        df['Lag_2'] = df['S&P500'].shift(2)
        df['Lag_3'] = df['S&P500'].shift(3)
        df['Lag_4'] = df['S&P500'].shift(4)

        # Drop rows with NaN values resulting from shifts
        df = df.dropna()

        feature_cols = ['S&P500', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4']
        self.X = df[feature_cols].values
        self.y = df['Target'].values

        print(f"Loaded and prepared {len(self.X)} samples for classification.")

    def _evaluate_and_plot(self, n_splits, title):
        if self.X is None or self.y is None:
            self.prepare_data()

        print(f"\n=== {title} ===")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # 1. K-Fold Accuracy Scores
        scores = cross_val_score(model, self.X, self.y, cv=kf, scoring='accuracy')
        print(f"Accuracy across {n_splits} folds: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
        print(f"Standard Deviation: {scores.std():.4f}")

        # 2. Out-of-fold predictions for Confusion Matrix
        y_pred = cross_val_predict(model, self.X, self.y, cv=kf)
        cm = confusion_matrix(self.y, y_pred)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{title} Results", fontsize=16)

        # Subplot 1: Bar chart of fold accuracies
        axes[0].bar(range(1, n_splits + 1), scores, color='skyblue', edgecolor='black')
        axes[0].axhline(y=scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.2f}')
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Accuracy per Fold ({n_splits} Splits)')
        axes[0].set_xticks(range(1, n_splits + 1))
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Subplot 2: Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down (0)", "Up (1)"])
        disp.plot(ax=axes[1], cmap='Blues', colorbar=True)
        axes[1].set_title('Overall Confusion Matrix (Cross-Validated)')

        plt.tight_layout()
        plt.show()

    def run_3_fold_classification(self):
        """Run 3-fold cross validation on the dataset and plot results."""
        self._evaluate_and_plot(n_splits=3, title="3-Fold Classification")

    def n_fold_classification(self, n=5):
        """
        Run n-fold cross validation and plot results. Default n is 5 as it fits best 
        to balance bias and variance for most datasets.
        """
        self._evaluate_and_plot(n_splits=n, title=f"{n}-Fold Classification")


# --- Execute when running this file directly ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "datasets", "sp500_index.csv")

    k_fold_obj = K_fold_sp500(csv_path)

    # 1. 3-fold classification
    # k_fold_obj.run_3_fold_classification()

    # 2. n-fold classification
    # Utilizing n=5 as it's conventionally the best fit for datasets of this size.
    # Note: Commented out so it does not run at the same time as 3-fold classification.
    # Uncomment the line below to run it separately.
    k_fold_obj.n_fold_classification(n=5)
