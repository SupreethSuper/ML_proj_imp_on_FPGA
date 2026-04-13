import os
import numpy as np
import matplotlib.pyplot as plt
from file_reader_csv import SP500Reader


class nquist_sampler:
    """Resample S&P500 data at 8× the Nyquist rate (i.e. 8× the max
    frequency found via FFT)."""

    def __init__(self, filepath: str, oversample_factor: int = 5):
        """
        Parameters
        ----------
        filepath : str
            Path to sp500_index.csv.
        oversample_factor : int
            Multiplier applied to the max frequency for the sampling rate
            (default 8).
        """
        self.oversample_factor = oversample_factor
        self.reader = SP500Reader(filepath)
        self.df = None
        self.signal = None
        self.abs_signal = None
        self.fft_result = None
        self.fft_freqs = None
        self.fft_magnitudes = None
        self.max_freq = None
        self.max_freq_magnitude = None
        self.sampling_rate = None
        self.sampled_indices = None
        self.sampled_signal = None
        self.sampled_dates = None

    def load_data(self):
        """Load CSV and extract the S&P500 column."""
        self.df = self.reader.get_data()
        self.signal = self.df["S&P500"].values
        # Convert all data values to absolute before FFT
        self.abs_signal = np.abs(self.signal)
        print(f"Loaded {len(self.signal)} data points.")
        return self.signal

    def find_max_frequency(self):
        """Run FFT on the absolute-valued signal and identify the dominant
        (max-magnitude) frequency."""
        if self.abs_signal is None:
            self.load_data()

        N = len(self.abs_signal)
        self.fft_result = np.fft.fft(self.abs_signal)
        self.fft_magnitudes = np.abs(self.fft_result)
        self.fft_freqs = np.fft.fftfreq(N)

        # Consider only positive, non-DC frequencies
        pos_mask = self.fft_freqs > 0
        pos_freqs = self.fft_freqs[pos_mask]
        pos_mags = self.fft_magnitudes[pos_mask]

        # Dominant frequency = positive freq with highest magnitude
        max_idx = np.argmax(pos_mags)
        self.max_freq = pos_freqs[max_idx]
        self.max_freq_magnitude = pos_mags[max_idx]

        print(f"Max frequency (dominant): {self.max_freq:.6f} cycles/sample")
        print(f"Max frequency magnitude : {self.max_freq_magnitude:.4f}")
        return self.max_freq

    def compute_sampling(self):
        """Compute the Nyquist-based sampling rate and resample the signal."""
        if self.max_freq is None:
            self.find_max_frequency()

        # Nyquist rate = 2 × max_freq; we sample at oversample_factor × max_freq
        self.sampling_rate = self.oversample_factor * self.max_freq

        # Sampling interval (in original sample units)
        sample_interval = int(round(1.0 / self.sampling_rate))
        if sample_interval < 1:
            sample_interval = 1

        N = len(self.signal)
        self.sampled_indices = np.arange(0, N, sample_interval)
        self.sampled_signal = self.signal[self.sampled_indices]
        self.sampled_dates = self.df["Date"].values[self.sampled_indices]

        print(
            f"Nyquist sampling rate: {self.sampling_rate:.6f} samples/sample-unit "
            f"(interval = {sample_interval})"
        )
        print(
            f"Resampled to {len(self.sampled_signal)} points "
            f"(from {N} original)."
        )
        return self.sampled_signal

    def plot_results(self):
        """Plot FFT magnitude spectrum and original vs. resampled signal."""
        if self.sampled_signal is None:
            self.compute_sampling()

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # --- Plot 1: FFT Magnitude Spectrum (positive frequencies) ---
        N = len(self.abs_signal)
        half = N // 2
        pos_freqs = self.fft_freqs[:half]
        pos_mags = self.fft_magnitudes[:half]

        axes[0].plot(pos_freqs, pos_mags, color="darkorange", linewidth=0.8)
        # Mark the max-frequency point
        axes[0].axvline(
            x=self.max_freq, color="red", linestyle="--", linewidth=1,
            label=f"Max freq = {self.max_freq:.6f} (mag = {self.max_freq_magnitude:.2f})",
        )
        axes[0].plot(
            self.max_freq, self.max_freq_magnitude, "rv", markersize=10,
        )
        axes[0].set_title("FFT Magnitude Spectrum (absolute-valued S&P500)")
        axes[0].set_xlabel("Frequency (cycles / sample)")
        axes[0].set_ylabel("Magnitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Plot 2: Original vs. Nyquist-resampled signal ---
        axes[1].plot(
            self.df["Date"],
            self.signal,
            color="steelblue",
            linewidth=0.6,
            alpha=0.5,
            label="Original S&P500",
        )
        axes[1].plot(
            self.sampled_dates,
            self.sampled_signal,
            "o-",
            color="crimson",
            markersize=2,
            linewidth=0.8,
            label=f"Nyquist Resampled ({self.oversample_factor}× max freq)",
        )

        axes[1].set_title(
            f"S&P500 — Nyquist Resampling at {self.oversample_factor}× Max Frequency"
        )
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Index Value")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run(self):
        """Full pipeline: load → find max freq → resample → plot."""
        self.load_data()
        self.find_max_frequency()
        self.compute_sampling()
        self.plot_results()

    def compare_factors(self, factors=None):
        """Compare multiple oversample factors and find the best one.

        Parameters
        ----------
        factors : list[int], optional
            List of oversample factors to compare (default [2..8]).

        Returns
        -------
        dict
            Best result with keys: factor, sampling_rate, n_samples, rmse.
        """
        if factors is None:
            factors = list(range(2, 9))

        # Ensure data is loaded and max frequency is found
        if self.abs_signal is None:
            self.load_data()
        if self.max_freq is None:
            self.find_max_frequency()

        original = self.signal
        N = len(original)
        results = []  # (factor, sampling_rate, n_samples, rmse)

        print("\n" + "=" * 65)
        print(f"  Comparing oversample factors {factors[0]} – {factors[-1]}")
        print("=" * 65)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot original on the sampler subplot
        axes[0].plot(
            self.df["Date"], original,
            color="steelblue", linewidth=0.6, alpha=0.4, label="Original S&P500",
        )

        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(factors)))

        for i, factor in enumerate(factors):
            # Compute sampling for this factor
            sampling_rate = factor * self.max_freq
            sample_interval = int(round(1.0 / sampling_rate))
            if sample_interval < 1:
                sample_interval = 1

            sampled_indices = np.arange(0, N, sample_interval)
            sampled_signal = original[sampled_indices]
            sampled_dates = self.df["Date"].values[sampled_indices]

            # Interpolate resampled signal back to original length for RMSE
            reconstructed = np.interp(
                np.arange(N), sampled_indices, sampled_signal
            )
            rmse = np.sqrt(np.mean((original - reconstructed) ** 2))

            results.append((factor, sampling_rate, len(sampled_signal), rmse))

            print(
                f"{factor}× | rate={sampling_rate:.6f} | "
                f"interval={sample_interval} | "
                f"n_samples={len(sampled_signal)} | RMSE={rmse:.4f}"
            )

            axes[0].plot(
                sampled_dates, sampled_signal, "o-",
                color=colors[i], markersize=3, linewidth=0.9,
                label=f"{factor}× (n={len(sampled_signal)}, RMSE={rmse:.2f})",
            )

        axes[0].set_title(f"S&P500 — Nyquist Resampling Comparison ({factors[0]}×–{factors[-1]}×)")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Index Value")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # --- Bar chart of RMSE per factor ---
        factor_labels = [str(r[0]) + "×" for r in results]
        rmse_vals = [r[3] for r in results]

        bars = axes[1].bar(factor_labels, rmse_vals, color=colors, edgecolor="black")
        axes[1].set_title("Reconstruction RMSE by Oversample Factor")
        axes[1].set_xlabel("Oversample Factor")
        axes[1].set_ylabel("RMSE")
        axes[1].grid(True, axis="y", alpha=0.3)

        # Annotate bar values
        for bar, val in zip(bars, rmse_vals):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
            )

        plt.tight_layout()
        plt.show()

        # --- Summary table ---
        print(f"\n{'Factor':<8}{'Sampling Rate':<22}{'# Samples':<12}{'RMSE':<12}")
        print("-" * 54)
        for factor, sr, ns, rmse in results:
            print(f"{factor}×{'':<5}{sr:<22.6f}{ns:<12}{rmse:<12.4f}")

        best = min(results, key=lambda r: r[3])
        print(f"\n>>> Best oversample factor : {best[0]}×")
        print(f">>> Best sampling frequency: {best[1]:.6f} samples/sample-unit")
        print(f">>> Lowest RMSE            : {best[3]:.4f}")

        return {
            "factor": best[0],
            "sampling_rate": best[1],
            "n_samples": best[2],
            "rmse": best[3],
        }


# --- Execute when running this file directly ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "datasets", "sp500_index.csv")

    # --- Single run (fixed sampling) ---
    sampler = nquist_sampler(csv_path)
    sampler.run()

    # # --- Compare factors 2–8 and find best sampling frequency ---
    # best = sampler.compare_factors(factors=list(range(2, 9)))
    # print(best)
