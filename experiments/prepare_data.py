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
    """Load and align all three source files via inner join."""
    # Uncertainty
    unc_path = os.path.join(data_dir, "Uncertanty_index_data_23_07.xlsx")
    df_unc = pd.read_excel(unc_path, sheet_name="Copy", header=0, skiprows=[1])
    df_unc = df_unc.rename(columns={"Unnamed: 0": "Date"}).set_index("Date")
    df_unc.index = pd.to_datetime(df_unc.index)

    # CDS 
    cds_path = os.path.join(data_dir, "CDS Poland.xlsx")
    
    # header=2 skips the first two metadata rows and uses row 3 as column headers
    df_cds = pd.read_excel(cds_path, header=2)
    
    # Rename the columns to match what the rest of your script expects
    df_cds = df_cds.rename(columns={
        "Timestamp": "Date",
        "MID_SPREAD": "POLAND CDS USD SR 5Y Corp"
    })
    
    # Drop any empty rows (like those created by empty columns A and B)
    df_cds = df_cds.dropna(subset=["Date"])
    
    # Parse the dates (dayfirst=True handles the DD.MM.YYYY format shown in your file)
    df_cds["Date"] = pd.to_datetime(df_cds["Date"], dayfirst=True)
    df_cds = df_cds.set_index("Date")

    # ASS
    ass_path = os.path.join(data_dir, "ASS.xlsx")
    excel_file = pd.ExcelFile(ass_path)
    sheet_names = excel_file.sheet_names

    # Match sheet names dynamically regardless of minor typos or trailing spaces
    spread_sheet = next((s for s in sheet_names if "swap" in s.lower()), sheet_names[0])
    bond_sheet = next((s for s in sheet_names if "bond" in s.lower() or "10-year" in s.lower()), sheet_names[1])

    # 1. Load Asset Swap Spread sheet
    df_ass_spread = pd.read_excel(excel_file, sheet_name=spread_sheet)
    df_ass_spread = df_ass_spread.rename(columns={
        "Unnamed: 0": "Date", 
        "Unnamed: 3": "Calculated Spread"
    })
    df_ass_spread = df_ass_spread.dropna(subset=["Date"])
    df_ass_spread["Date"] = pd.to_datetime(df_ass_spread["Date"], dayfirst=True)
    df_ass_spread = df_ass_spread.set_index("Date")

    # 2. Load Poland 10-Year Bond Yield Histo sheet
    df_ass_bond = pd.read_excel(excel_file, sheet_name=bond_sheet)
    df_ass_bond = df_ass_bond.rename(columns={
        "Data": "Date", 
        "Ostatnio": "GTPLN10Y Govt.1"  
    })
    df_ass_bond = df_ass_bond.dropna(subset=["Date"])
    df_ass_bond["Date"] = pd.to_datetime(df_ass_bond["Date"], dayfirst=True)
    df_ass_bond = df_ass_bond.set_index("Date")

    # Combine internal ASS sheets
    df_ass = df_ass_spread.join(df_ass_bond, how="outer")

    # Inner join resolves Polish/US/German holiday misalignment natively
    df = df_cds.join([df_ass, df_unc], how="inner")
    
    return df.sort_index()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select target + curated features, forward-fill to business days."""
    cols = [TARGET_COL] + [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        print(f"[warn] columns not found in source file, skipping: {missing}")
    frame = df[cols].copy()
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
