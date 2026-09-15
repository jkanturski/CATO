"""
prepare_data.py — feature engineering pipeline for Poland CDS 5Y forecasting.

Reads the three Bloomberg/Refinitiv source files from CATO/data, builds
the curated feature set described in paper/paper.tex Section III/IV, and
writes train/val/test parquet files for each forecasting horizon
(h in {1, 7, 30} trading days).

Usage:
    python prepare_data.py --data-dir /home/jkanturski/CATO/data \
                            --out-dir /home/jkanturski/CATO/experiments/data

Requires: pandas, numpy, openpyxl.
"""
import argparse
import os

import numpy as np
import pandas as pd

TARGET_COL = "POLAND CDS USD SR 5Y Corp"

# Curated feature set (Section IV.B / Table I of paper.tex)
FEATURE_COLS = [
    "POLAND CDS USD SR 2Y Corp",
    "POLAND CDS USD SR 10Y Corp",
    "GERMAN CDS USD SR 5Y Corp",
    "GTPLN10Y Govt.1",
    "GTDEM10Y Govt",
    "Spread DE PL 10y",
    "Spread DE PL 5y",
    "WIBR3M Index",
    "WIRON Index",
    "MOVE Index",
    "VIX Index",
    "V2X Index",
    "WIG20 Index",
    "EURPLN Curncy",
    "USDPLN Curncy",
    "EPUCGLCP Index",
    "GPRXGPRD Index",
    "GPRXMA30 Index",
]

LAGS = [1, 5, 20]
HORIZONS = [1, 7, 30]

def load_raw(data_dir: str) -> pd.DataFrame:
    """Load the master Uncertainty Index dataset containing all features and target."""
    unc_path = os.path.join(data_dir, "Uncertanty_index_data_23_07.xlsx")
    
    # header=0 reads Excel Row 1 as column names; skiprows=[1] drops Excel Row 2 ("Last Price")
    df_unc = pd.read_excel(unc_path, sheet_name="Copy", header=0, skiprows=[1])
    
    # Clean all column headers aggressively
    df_unc.columns = df_unc.columns.astype(str).str.replace(r'[\n\r\xa0]+', ' ', regex=True)
    df_unc.columns = df_unc.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Remove any duplicate column headers loaded from Excel
    df_unc = df_unc.loc[:, ~df_unc.columns.duplicated()]
    
    # Set up Date index from Column A (index 0)
    date_col = df_unc.columns[0]
    df_unc = df_unc.rename(columns={date_col: "Date"})
    df_unc = df_unc.dropna(subset=["Date"])
    df_unc["Date"] = pd.to_datetime(df_unc["Date"], dayfirst=True, errors="coerce").dt.normalize()
    df_unc = df_unc.dropna(subset=["Date"])
    
    # Deduplicate dates before setting index
    df_unc = df_unc.drop_duplicates(subset=["Date"], keep="last")
    df_unc = df_unc.set_index("Date")

    # Drop unneeded metadata or unnamed columns
    unnamed = [c for c in df_unc.columns if "Unnamed:" in str(c)]
    df_unc.drop(columns=unnamed, inplace=True, errors="ignore")

    return df_unc.sort_index()
    

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select target + curated features, forward-fill to business days."""
    
    if TARGET_COL not in df.columns:
        print(f"\n[FATAL ERROR] Target column '{TARGET_COL}' not found.")
        raise KeyError(f"Target '{TARGET_COL}' not in index")

    cols = [TARGET_COL] + [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        print(f"[warn] columns not found in source file, skipping: {missing}")
        
    frame = df[cols].copy()

    frame = frame[~frame.index.duplicated(keep='last')]
    
    frame = frame.asfreq("B")  # business-day frequency
    frame = frame.ffill()
    return frame


def add_lags(frame: pd.DataFrame, feature_cols) -> pd.DataFrame:
    out = frame.copy()
    for col in feature_cols:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = frame[col].shift(lag)
    return out


def add_targets(frame: pd.DataFrame, target_col: str, horizons) -> pd.DataFrame:
    out = frame.copy()
    for h in horizons:
        out[f"target_h{h}"] = frame[target_col].shift(-h)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/jkanturski/CATO/data")
    parser.add_argument(
        "--out-dir",
        default="/home/jkanturski/CATO/experiments/data",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw = load_raw(args.data_dir)
    frame = build_feature_frame(raw)

    # Restrict to the window where the target (CDS 5Y) is available.
    frame = frame.dropna(subset=[TARGET_COL])
    print(f"CDS 5Y available from {frame.index.min()} to {frame.index.max()}, "
          f"{len(frame)} business days.")

    feature_cols = [c for c in FEATURE_COLS if c in frame.columns]
    lagged = add_lags(frame, feature_cols)
    full = add_targets(lagged, TARGET_COL, HORIZONS)
    
    # Drop rows at the tail where the h=30 shift created NaNs
    target_cols = [f"target_h{h}" for h in HORIZONS]
    full = full.dropna(subset=target_cols)

    # Standardize features on train split only (2013-2022), applied to all.
    train_mask = (full.index >= "2013-01-01") & (full.index <= "2022-12-31")
    val_mask = (full.index >= "2023-01-01") & (full.index <= "2023-12-31")
    test_mask = full.index >= "2024-01-01"

    # Scale BOTH the base features and their lags to prevent exploding gradients
    cols_to_scale = feature_cols + [c for c in full.columns if "_lag" in c]
    
    mu = full.loc[train_mask, cols_to_scale].mean()
    sigma = full.loc[train_mask, cols_to_scale].std().replace(0, 1.0)
    full[cols_to_scale] = (full[cols_to_scale] - mu) / sigma
    
    full.loc[train_mask].to_parquet(os.path.join(args.out_dir, "train.parquet"))
    full.loc[val_mask].to_parquet(os.path.join(args.out_dir, "val.parquet"))
    full.loc[test_mask].to_parquet(os.path.join(args.out_dir, "test.parquet"))

    print(f"Wrote train ({train_mask.sum()}), val ({val_mask.sum()}), "
          f"test ({test_mask.sum()}) rows to {args.out_dir}")


if __name__ == "__main__":
    main()
