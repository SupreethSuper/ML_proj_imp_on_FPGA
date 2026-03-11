import pandas as pd


class SP500Reader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self) -> pd.DataFrame:
        """Read the CSV and return a DataFrame with Date and S&P 500 columns."""
        self.data = pd.read_csv(
            self.filepath,
            usecols=["Date", "S&P 500"],
            parse_dates=["Date"]
        )
        return self.data

    def get_data(self) -> pd.DataFrame:
        """Return the loaded data, loading it first if not already done."""
        if self.data is None:
            self.load()
        return self.data


# --- Quick test when running this file directly ---
if __name__ == "__main__":
    reader = SP500Reader(r"datasets\sp500_index.csv")  # <-- Change to your file path
    df = reader.get_data()
    print(df.head())