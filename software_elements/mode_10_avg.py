import os
import numpy as np
import matplotlib.pyplot as plt
from file_reader_csv import SP500Reader


class mode_avg:
    """Compute a mode-10 average (average every 10 consecutive data points)
    on the S&P500 index data."""

    def __init__(self, filepath: str, window: int = 50):
        """
        Parameters
        ----------
        filepath : str
            Path to sp500_index.csv.
        window : int
            Number of consecutive points to average (default 10).
        """
        self.filepath = filepath
        self.window = window
        self.reader = SP500Reader(filepath)
        self.df = None
        self.signal = None
        self.averaged = None
        self.averaged_dates = None

    def load_data(self):
        """Load the CSV and extract the S&P500 column."""
        self.df = self.reader.get_data()
        self.signal = self.df["S&P500"].values
        print(f"Loaded {len(self.signal)} data points.")
        return self.signal

    def compute_mode_avg(self):
        """Average every `window` consecutive data points."""
        if self.signal is None:
            self.load_data()

        N = len(self.signal)
        # Number of complete groups of `window`
        n_groups = N // self.window
        trimmed = self.signal[: n_groups * self.window]

        self.averaged = trimmed.reshape(n_groups, self.window).mean(axis=1)

        # Pick the midpoint date of each group for plotting
        dates = self.df["Date"].values
        trimmed_dates = dates[: n_groups * self.window]
        mid_indices = np.arange(self.window // 2, n_groups * self.window, self.window)
        self.averaged_dates = trimmed_dates[mid_indices]

        print(
            f"Mode-{self.window} average: {n_groups} groups "
            f"({N - n_groups * self.window} trailing points dropped)."
        )
        return self.averaged

    def plot_results(self):
        """Plot original signal vs. mode-10 averaged signal."""
        if self.averaged is None:
            self.compute_mode_avg()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            self.df["Date"],
            self.signal,
            color="steelblue",
            linewidth=0.6,
            alpha=0.5,
            label="Original S&P500",
        )
        ax.plot(
            self.averaged_dates,
            self.averaged,
            color="crimson",
            linewidth=1.2,
            label=f"Mode-{self.window} Average",
        )

        ax.set_title(f"S&P500 Index — Mode-{self.window} Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run(self):
        """Full pipeline: load → compute → plot."""
        self.load_data()
        self.compute_mode_avg()
        self.plot_results()


# --- Execute when running this file directly ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "datasets", "sp500_index.csv")

    avg_obj = mode_avg(csv_path)
    avg_obj.run()
