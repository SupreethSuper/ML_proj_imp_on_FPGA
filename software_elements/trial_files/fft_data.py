import os
import numpy as np
import matplotlib.pyplot as plt
from file_reader_csv import SP500Reader


class fft__data:
    """Perform FFT analysis on S&P500 index data."""

    def __init__(self, filepath: str):
        """
        Initialize with the path to the S&P500 CSV file.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to sp500_index.csv.
        """
        self.filepath = filepath
        self.reader = SP500Reader(filepath)
        self.df = None
        self.signal = None
        self.fft_result = None
        self.fft_freq = None
        self.fft_magnitude = None

    def load_data(self):
        """Load the CSV and extract the S&P500 column as the signal."""
        self.df = self.reader.get_data()
        self.signal = self.df["S&P500"].values
        print(f"Loaded {len(self.signal)} data points.")
        return self.signal

    def compute_fft(self):
        """Compute the FFT of the S&P500 signal."""
        if self.signal is None:
            self.load_data()

        N = len(self.signal)
        self.fft_result = np.fft.fft(self.signal)
        self.fft_freq = np.fft.fftfreq(N)
        self.fft_magnitude = np.abs(self.fft_result)
        print("FFT computation complete.")
        return self.fft_result

    def plot_results(self):
        """Plot the original signal and its FFT magnitude spectrum."""
        if self.fft_result is None:
            self.compute_fft()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # --- Original S&P500 signal ---
        axes[0].plot(self.df["Date"], self.signal, color="steelblue", linewidth=0.8)
        axes[0].set_title("S&P500 Index — Time Domain")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Index Value")
        axes[0].grid(True, alpha=0.3)

        # --- FFT magnitude (positive frequencies only) ---
        N = len(self.signal)
        half = N // 2
        axes[1].plot(
            self.fft_freq[:half],
            self.fft_magnitude[:half],
            color="crimson",
            linewidth=0.8,
        )
        axes[1].set_title("FFT Magnitude Spectrum (Positive Frequencies)")
        axes[1].set_xlabel("Frequency (cycles / sample)")
        axes[1].set_ylabel("Magnitude")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run(self):
        """Full pipeline: load → FFT → plot."""
        self.load_data()
        self.compute_fft()
        self.plot_results()


# --- Execute when running this file directly ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "datasets", "sp500_index.csv")

    fft_obj = fft__data(csv_path)
    fft_obj.run()
