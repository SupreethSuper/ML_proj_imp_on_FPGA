"""
sp500_60day_window.py
==============================
Load SP500V3.xlsx and extract a 60-day window from 04-04-2016 to 27-06-2016.

This script reads the xlsx file (not CSV) and filters the data for the specified
date range, then provides basic statistics and optional exports.
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# 1. Load Data from XLSX
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "SP500V3.xlsx")
SHEET = "SP500"

print(f"Reading sheet '{SHEET}' from '{DATA_PATH}' ...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")

print(f"Total rows loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================
# 2. Filter 60-Day Window (04-04-2016 to 27-06-2016)
# ============================================================
# Ensure date column is parsed as datetime
if "observation_date" in df.columns:
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    date_col = "observation_date"
else:
    # Try other common date column names
    date_cols = [col for col in df.columns if "date" in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        raise ValueError("No date column found in the spreadsheet")

# Define the 60-day window
start_date = pd.to_datetime("2016-04-04")
end_date = pd.to_datetime("2016-06-27")

df_window = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
df_window = df_window.sort_values(by=date_col).reset_index(drop=True)

# ============================================================
# 3. Display Window Statistics
# ============================================================
print("=" * 70)
print(f"60-DAY WINDOW: {start_date.date()} to {end_date.date()}")
print("=" * 70)
print(f"Rows in window: {len(df_window)}")
print(f"Date range in data: {df_window[date_col].min().date()} to {df_window[date_col].max().date()}")
print(f"Number of days: {(df_window[date_col].max() - df_window[date_col].min()).days + 1}")

if len(df_window) > 0:
    print(f"\nFirst row:\n{df_window.iloc[0]}")
    print(f"\nLast row:\n{df_window.iloc[-1]}")
    
    # Basic statistics on numeric columns
    numeric_cols = df_window.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Sample numeric summary:\n{df_window[numeric_cols].describe()}")
else:
    print("\n⚠ WARNING: No data found in the specified date range!")

# ============================================================
# 4. Optional: Export Window to CSV
# ============================================================
export_path = os.path.join(os.path.dirname(__file__), "sp500_window_2016_04_06_27.csv")
df_window.to_csv(export_path, index=False)
print(f"\n✓ Window data exported to: {export_path}")

# ============================================================
# 5. Summary
# ============================================================
print("\n" + "=" * 70)
print(f"✓ Successfully loaded 60-day window with {len(df_window)} records")
print("=" * 70)
