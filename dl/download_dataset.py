import kagglehub
import polars as pl
import yfinance as yf
import os

# 1. Download the dataset and get the cache directory
print("Downloading Solana dataset from Kaggle...")
dataset_dir = kagglehub.dataset_download("craigdagama/solana-historical-data")

# 2. Find the CSV file inside the downloaded directory
csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")
    
sol_csv_path = os.path.join(dataset_dir, csv_files[0])
print(f"Loading data from: {sol_csv_path}")

# 3. Load and format the Solana data
sol_raw = pl.read_csv(sol_csv_path)

# (If the dataset has dirty column names, adjust "Date", "Close", "Volume" accordingly)
sol_df = sol_raw.select([
    pl.col("Date").str.to_datetime().dt.replace_time_zone(None).alias("timestamp"),
    pl.col("Close").cast(pl.Float64).alias("sol_close"),
    pl.col("Volume").cast(pl.Float64).alias("sol_volume"),
])

# 4. Fetch matching Aave data via yfinance
print("Fetching matching Aave data...")
start_date = sol_df["timestamp"].min().strftime("%Y-%m-%d")
end_date = sol_df["timestamp"].max().strftime("%Y-%m-%d")

aave_pd = yf.Ticker("AAVE-USD").history(start=start_date, end=end_date).reset_index()
aave_df = pl.from_pandas(aave_pd).select([
    pl.col("Date").dt.cast_time_unit("ms").dt.replace_time_zone(None).alias("timestamp"),
    pl.col("Close").alias("aave_close"),
    pl.col("Volume").alias("aave_volume"),
])

# 5. Merge and calculate returns
print("Merging datasets...")
merged_df = (
    sol_df.join(aave_df, on="timestamp", how="inner")
    .sort("timestamp")
    .with_columns([
        (pl.col("sol_close") / pl.col("sol_close").shift(1) - 1).alias("sol_return"),
        (pl.col("aave_close") / pl.col("aave_close").shift(1) - 1).alias("aave_return"),
    ])
    .drop_nulls()
)

# 6. Save directly to the shared data directory
output_path = "data/train_solana_aave.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.write_parquet(output_path)

print(f"Success! {len(merged_df)} rows saved to {output_path}")
