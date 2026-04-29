import os
import pandas as pd


class CSVReader:
    """General-purpose CSV reader that loads any CSV file into a DataFrame."""

    def __init__(self, filepath: str, usecols=None, parse_dates=None):
        self.filepath = filepath
        self.usecols = usecols
        self.parse_dates = parse_dates
        self.data = None

    def load(self) -> pd.DataFrame:
        """Read the CSV and return a DataFrame."""
        self.data = pd.read_csv(
            self.filepath,
            usecols=self.usecols,
            parse_dates=self.parse_dates if self.parse_dates else False,
        )
        return self.data

    def get_data(self) -> pd.DataFrame:
        """Return the loaded data, loading it first if not already done."""
        if self.data is None:
            self.load()
        return self.data


class SP500Reader(CSVReader):
    """Convenience reader pre-configured for the S&P500 index CSV."""

    def __init__(self, filepath: str):
        super().__init__(
            filepath,
            usecols=["Date", "S&P500"],
            parse_dates=["Date"],
        )


# --- Quick test when running this file directly ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "datasets", "sp500_index.csv")
    reader = SP500Reader(csv_path)
    df = reader.get_data()
    print(df.head())
