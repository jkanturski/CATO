import os
import kagglehub
import pandas as pd
import requests

# 1. Download Solana dataset from Kaggle
print("Downloading Solana dataset from Kaggle...")
dataset_dir = kagglehub.dataset_download("craigdagama/solana-historical-data")

csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the Kaggle download.")

sol_csv_path = os.path.join(dataset_dir, csv_files[0])
print(f"Loading Solana data from: {sol_csv_path}")

sol_df = pd.read_csv(sol_csv_path)

# Normalize column names: remove whitespace and make lowercase
sol_df.columns = sol_df.columns.str.strip().str.lower()

sol_df['timestamp'] = pd.to_datetime(
    sol_df['Day'],
    dayfirst=True
).dt.tz_localize(None)

sol_df['sol_close'] = sol_df['close'].astype(float)
sol_df['sol_volume'] = sol_df['volume'].astype(float)
sol_df = sol_df[['timestamp', 'sol_close', 'sol_volume']].sort_values('timestamp')

# 2. Fetch Aave historical data directly from Binance REST API 
print("Fetching Aave historical data from Binance API...")
url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "AAVEUSDT",
    "interval": "1d",
    "limit": 1000
}
response = requests.get(url, params=params)
response.raise_for_status()

# Format: [open_time, open, high, low, close, volume, ...]
aave_records = []
for kline in response.json():
    aave_records.append({
        "timestamp": pd.to_datetime(kline[0], unit='ms'),
        "aave_close": float(kline[4]),
        "aave_volume": float(kline[5])
    })
aave_df = pd.DataFrame(aave_records)

# 3. Merge and generate return features
print("Merging datasets...")
merged_df = pd.merge(sol_df, aave_df, on="timestamp", how="inner")
merged_df['sol_return'] = merged_df['sol_close'].pct_change()
merged_df['aave_return'] = merged_df['aave_close'].pct_change()
merged_df = merged_df.dropna().reset_index(drop=True)

# 4. Save to shared parquet path
output_path = "data/train_solana_aave.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_parquet(output_path)

print(f"Successfully created {output_path} with {len(merged_df)} records.")
