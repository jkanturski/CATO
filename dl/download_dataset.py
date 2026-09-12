import os
import kagglehub
import pandas as pd
import requests
import time


# ============================================================
# 1. Download Solana dataset from Kaggle
# ============================================================

print("Downloading Solana dataset from Kaggle...")

dataset_dir = kagglehub.dataset_download(
    "craigdagama/solana-historical-data"
)

csv_files = [
    f for f in os.listdir(dataset_dir)
    if f.endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found in the Kaggle download."
    )

sol_csv_path = os.path.join(
    dataset_dir,
    csv_files[0]
)

print(f"Loading Solana data from: {sol_csv_path}")

sol_df = pd.read_csv(sol_csv_path)


# ------------------------------------------------------------
# Normalize column names
# ------------------------------------------------------------

sol_df.columns = (
    sol_df.columns
    .str.strip()
    .str.lower()
)

print("Solana columns:", sol_df.columns.tolist())


# ------------------------------------------------------------
# Parse Solana date
# Original format: DD/MM/YYYY
# ------------------------------------------------------------

sol_df["timestamp"] = pd.to_datetime(
    sol_df["day"],
    dayfirst=True,
    errors="coerce"
)


# ------------------------------------------------------------
# Numeric conversion
# ------------------------------------------------------------

sol_df["sol_close"] = pd.to_numeric(
    sol_df["close"],
    errors="coerce"
)

sol_df["sol_volume"] = pd.to_numeric(
    sol_df["volume"],
    errors="coerce"
)


# ------------------------------------------------------------
# Keep required columns
# ------------------------------------------------------------

sol_df = sol_df[
    [
        "timestamp",
        "sol_close",
        "sol_volume"
    ]
].dropna()


# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

sol_df = (
    sol_df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(
    f"Solana range: "
    f"{sol_df['timestamp'].min()} -> "
    f"{sol_df['timestamp'].max()}"
)

print(
    f"Solana records: {len(sol_df)}"
)


# ============================================================
# 2. Fetch complete AAVE historical data from Binance
# ============================================================

print("\nFetching AAVE historical data from Binance API...")

url = "https://api.binance.com/api/v3/klines"

start_time = int(
    sol_df["timestamp"].min().timestamp() * 1000
)

end_time = int(
    sol_df["timestamp"].max().timestamp() * 1000
)

interval = "1d"

aave_records = []

current_start = start_time


while current_start < end_time:

    params = {
        "symbol": "AAVEUSDT",
        "interval": interval,
        "startTime": current_start,
        "endTime": end_time,
        "limit": 1000
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    klines = response.json()

    if not klines:
        break

    for kline in klines:

        aave_records.append({
            "timestamp": pd.to_datetime(
                kline[0],
                unit="ms"
            ),
            "aave_close": float(kline[4]),
            "aave_volume": float(kline[5])
        })

    last_open_time = klines[-1][0]

    next_start = last_open_time + 1

    if next_start <= current_start:
        break

    current_start = next_start

    print(
        f"Downloaded {len(aave_records)} "
        f"AAVE candles..."
    )

    time.sleep(0.2)


aave_df = pd.DataFrame(aave_records)


# ------------------------------------------------------------
# Clean AAVE data
# ------------------------------------------------------------

aave_df = (
    aave_df
    .drop_duplicates(subset=["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(
    f"AAVE range: "
    f"{aave_df['timestamp'].min()} -> "
    f"{aave_df['timestamp'].max()}"
)

print(
    f"AAVE records: {len(aave_df)}"
)


# ============================================================
# 3. Merge Solana and AAVE
# ============================================================

print("\nMerging datasets...")

merged_df = pd.merge(
    sol_df,
    aave_df,
    on="timestamp",
    how="inner"
)

merged_df = (
    merged_df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# 4. Generate return features
# ============================================================

merged_df["sol_return"] = (
    merged_df["sol_close"]
    .pct_change()
)

merged_df["aave_return"] = (
    merged_df["aave_close"]
    .pct_change()
)


# Remove first row caused by pct_change()
merged_df = (
    merged_df
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# 5. Save Parquet
# ============================================================

output_path = "data/train_solana_aave.parquet"

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

merged_df.to_parquet(
    output_path,
    index=False
)


# ============================================================
# 6. Report
# ============================================================

print("\n========================================")
print("Dataset successfully created")
print("========================================")

print(f"Output: {output_path}")
print(f"Records: {len(merged_df)}")

print(
    f"Date range: "
    f"{merged_df['timestamp'].min()} -> "
    f"{merged_df['timestamp'].max()}"
)

print("\nColumns:")
print(merged_df.columns.tolist())

print("\nFirst rows:")
print(merged_df.head())

print("\nLast rows:")
print(merged_df.tail())
