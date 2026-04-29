"""
PCA (Principal Component Analysis) on S&P 500 stock data.

Reads multi-feature stock data via CSVReader from file_reader_csv.py,
applies PCA to reduce dimensionality, and visualises the results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from file_reader_csv import CSVReader


# ── helpers ──────────────────────────────────────────────────────────


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance standardization (column-wise).

    Returns:
        X_std: standardized data
        mean:  per-feature mean
        std:   per-feature standard deviation
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid division by zero for constant columns
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def compute_pca(
    X_std: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA via eigen-decomposition of the covariance matrix.

    Args:
        X_std:        standardized data matrix  (N x D)
        n_components: number of principal components to keep
                      (default: keep all)

    Returns:
        scores:            projected data              (N x n_components)
        eigenvectors:      principal-component axes    (D x n_components)
        explained_variance_ratio: fraction of variance per component
    """
    N = X_std.shape[0]
    # Covariance matrix  (D x D)
    cov_matrix = (X_std.T @ X_std) / (N - 1)

    # Eigen-decomposition (returns sorted ascending ─ flip to descending)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance ratio
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    # Keep only the requested number of components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

    # Project data onto principal components
    scores = X_std @ eigenvectors

    return scores, eigenvectors, explained_variance_ratio


# ── plotting ─────────────────────────────────────────────────────────


def plot_explained_variance(explained_variance_ratio: np.ndarray) -> None:
    """Bar + cumulative-line chart of explained variance."""
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), explained_variance_ratio, alpha=0.6,
           label="Individual")
    ax.step(range(1, n + 1), cumulative, where="mid", color="red",
            label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA – Explained Variance")
    ax.legend()
    ax.set_xticks(range(1, n + 1))
    plt.tight_layout()
    plt.show()


def plot_2d_projection(scores: np.ndarray,
                       explained_variance_ratio: np.ndarray) -> None:
    """Scatter plot of the first two principal components."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scores[:, 0], scores[:, 1], s=8, alpha=0.5)
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]:.2%} variance)")
    ax.set_title("PCA – 2-D Projection of S&P 500 Stock Data")
    plt.tight_layout()
    plt.show()


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    # ── 1. Load data using CSVReader ─────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "datasets", "sp500_stocks.csv")

    reader = CSVReader(csv_path)
    df = reader.get_data()
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # ── 2. Prepare feature matrix ────────────────────────────────────
    #   Pivot: rows = dates, columns = stock symbols, values = Close price
    #   This gives us one feature per stock → PCA finds co-movement patterns.
    numeric_col = "Close"
    pivot_df = df.pivot_table(index="Date", columns="Symbol",
                              values=numeric_col)

    # Drop stocks that have >50 % missing values, then fill remaining gaps
    threshold = len(pivot_df) * 0.5
    pivot_df = pivot_df.dropna(axis=1, thresh=int(threshold))
    pivot_df = pivot_df.ffill().bfill().dropna()

    print(f"Pivot table shape (dates × stocks): {pivot_df.shape}")
    stock_names = list(pivot_df.columns)

    X = pivot_df.values  # (N_dates x N_stocks)

    # ── 3. Standardize ──────────────────────────────────────────────
    X_std, mean, std = standardize(X)

    # ── 4. Run PCA ──────────────────────────────────────────────────
    n_components = min(10, X_std.shape[1])  # keep up to 10 PCs
    scores, eigvecs, evr = compute_pca(X_std, n_components=n_components)

    print(f"\nExplained variance ratio (top {n_components} components):")
    for i, v in enumerate(evr, 1):
        print(f"  PC{i}: {v:.4%}")
    print(f"  Total: {evr.sum():.4%}\n")

    # ── 5. Show top-contributing stocks for PC1 ─────────────────────
    pc1_loadings = eigvecs[:, 0]
    top_k = 10
    top_idx = np.argsort(np.abs(pc1_loadings))[::-1][:top_k]
    print(f"Top {top_k} stocks contributing to PC1:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {stock_names[idx]:>5s}  loading = {pc1_loadings[idx]:+.4f}")

    # ── 6. Plot results ─────────────────────────────────────────────
    plot_explained_variance(evr)
    plot_2d_projection(scores, evr)


if __name__ == "__main__":
    main()
