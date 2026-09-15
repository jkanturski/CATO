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
    """Load and align all source files via inner join using dynamic header matching."""
    # 1. Uncertainty Index 
    unc_path = os.path.join(data_dir, "Uncertanty_index_data_23_07.xlsx")
    
    # header=3 tells Pandas to use Excel Row 4 as the column names (0-indexed)
    df_unc = pd.read_excel(unc_path, sheet_name="Copy", header=3)
    
    # Cell A4 is blank, so Pandas names it 'Unnamed: 0'. Rename it to 'Date'.
    df_unc = df_unc.rename(columns={df_unc.columns[0]: "Date"})
    
    # Excel Row 5 contains the string 'Dates' in column A. We need to filter it out.
    df_unc = df_unc[df_unc["Date"] != "Dates"]
    
    # Clean the headers just in case there are trailing spaces
    df_unc.columns = df_unc.columns.astype(str).str.strip()
    
    # Drop rows with missing dates, convert to datetime, and set the index
    df_unc = df_unc.dropna(subset=["Date"])
    df_unc["Date"] = pd.to_datetime(df_unc["Date"], errors='coerce', dayfirst=True)
    df_unc = df_unc.dropna(subset=["Date"]) # Drops any garbage rows that couldn't be parsed
    df_unc = df_unc.set_index("Date")

    # 2. CDS Poland
    cds_path = os.path.join(data_dir, "CDS Poland.xlsx")
    df_cds = pd.read_excel(cds_path, header=2)
    df_cds.columns = df_cds.columns.astype(str).str.strip()
    
    cds_date_col = next((c for c in df_cds.columns if "timestamp" in c.lower() or "date" in c.lower()), None)
    cds_spread_col = next((c for c in df_cds.columns if "mid_spread" in c.lower() or "spread" in c.lower()), None)
    
    if not cds_spread_col or not cds_date_col:
        raise KeyError(
            f"Could not locate CDS target columns in {cds_path}. "
            f"Detected headers: {list(df_cds.columns)}"
        )
        
    df_cds = df_cds.rename(columns={
        cds_date_col: "Date",
        cds_spread_col: "POLAND CDS USD SR 5Y Corp"
    })
    df_cds = df_cds.dropna(subset=["Date"])
    df_cds["Date"] = pd.to_datetime(df_cds["Date"], dayfirst=True)
    df_cds = df_cds.set_index("Date")

    # 3. ASS Data
    ass_path = os.path.join(data_dir, "ASS.xlsx")
    excel_file = pd.ExcelFile(ass_path)
    sheet_names = excel_file.sheet_names

    spread_sheet = next((s for s in sheet_names if "swap" in s.lower() or "spead" in s.lower()), sheet_names[0])
    bond_sheet = next((s for s in sheet_names if "bond" in s.lower() or "10-year" in s.lower()), sheet_names[1])

    # 3a. Asset Swap Spread Sheet
    df_ass_spread = pd.read_excel(excel_file, sheet_name=spread_sheet)
    df_ass_spread.columns = df_ass_spread.columns.astype(str).str.strip()
    
    ass_date_col = next((c for c in df_ass_spread.columns if "date" in c.lower() or "unnamed: 0" in c.lower()), df_ass_spread.columns[0])
    df_ass_spread = df_ass_spread.rename(columns={ass_date_col: "Date"})
    df_ass_spread = df_ass_spread.dropna(subset=["Date"])
    df_ass_spread["Date"] = pd.to_datetime(df_ass_spread["Date"], dayfirst=True)
    df_ass_spread = df_ass_spread.set_index("Date")

    # 3b. Poland 10-Year Bond Yield Histo Sheet
    df_ass_bond = pd.read_excel(excel_file, sheet_name=bond_sheet)
    df_ass_bond.columns = df_ass_bond.columns.astype(str).str.strip()
    
    bond_date_col = next((c for c in df_ass_bond.columns if "data" in c.lower() or "date" in c.lower()), None)
    bond_yield_col = next((c for c in df_ass_bond.columns if "ostatnio" in c.lower() or "close" in c.lower() or "mid" in c.lower()), None)
    
    if bond_date_col and bond_yield_col:
        df_ass_bond = df_ass_bond.rename(columns={
            bond_date_col: "Date",
            bond_yield_col: "GTPLN10Y Govt.1"
        })
    df_ass_bond = df_ass_bond.dropna(subset=["Date"])
    df_ass_bond["Date"] = pd.to_datetime(df_ass_bond["Date"], dayfirst=True)
    df_ass_bond = df_ass_bond.set_index("Date")

    # Combine internal ASS sheets
    df_ass = df_ass_spread.join(df_ass_bond, how="outer", rsuffix="_bond")

    # Clean up unneeded metadata columns before merging
    for frame in [df_cds, df_ass, df_unc]:
        unnamed = [c for c in frame.columns if "Unnamed:" in str(c)]
        frame.drop(columns=unnamed, inplace=True, errors="ignore")

    # Final inner join across all datasets
    df = df_cds.join([df_ass, df_unc], how="inner")
    return df.sort_index()
    

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select target + curated features, forward-fill to business days."""
    
    if TARGET_COL not in df.columns:
        print(f"\n[FATAL ERROR] Target column '{TARGET_COL}' not found.")
        raise KeyError(f"Target '{TARGET_COL}' not in index")

    cols = [TARGET_COL] + [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        print(f"[warn] columns not found in source file, skipping: {len(missing)} columns")
        
    frame = df[cols].copy()
    
    # --- ADD THIS LINE to remove duplicate dates ---
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
