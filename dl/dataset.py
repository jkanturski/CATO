import torch
import polars as pl
import numpy as np
from torch.utils.data import Dataset

class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, parquet_path, target_col="sol_return", seq_length=128, forecast_horizon=1):
        """
        Args:
            parquet_path (str): Path to parquet dataset file.
            target_col (str): Column name to use as prediction target ('sol_return' or 'aave_return').
            seq_length (int): Length of historical lookback window.
            forecast_horizon (int): Steps ahead to predict (1 = next day return).
        """
        self.df = pl.read_parquet(parquet_path)
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        if target_col not in self.df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found. "
                f"Available columns: {self.df.columns}"
            )

        # Separate target column and features (exclude timestamp string/datetime)
        feature_cols = [c for c in self.df.columns if c != "timestamp"]
        
        features_np = self.df.select(feature_cols).to_numpy().astype(np.float32)
        target_idx = feature_cols.index(target_col)

        # Pre-convert full dataset into PyTorch Tensors for zero-copy slicing
        self.features = torch.from_numpy(features_np)
        self.targets = self.features[:, target_idx]

    def __len__(self):
        # Subtract sequence length and forecast steps to prevent index out of bounds
        return len(self.features) - self.seq_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # Sliding input window: [idx : idx + seq_length]
        x = self.features[idx : idx + self.seq_length]
        
        # Target for next step: (idx + seq_length - 1 + forecast_horizon)
        target_pos = idx + self.seq_length - 1 + self.forecast_horizon
        y = self.targets[target_pos]
        
        return x, y
