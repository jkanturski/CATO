"""
dataset.py — sliding-window sequence dataset for LSTM/TCN training on
Poland CDS 5Y forecasting.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CDSSequenceDataset(Dataset):
    """Produces (X, y) pairs where X is a (lookback, n_features) window of
    raw (unlagged) standardized features and y is the target CDS 5Y value
    h trading days ahead, for a single horizon h."""

    def __init__(self, parquet_path, feature_cols, horizon, lookback=60):
        self.df = pd.read_parquet(parquet_path)
        self.feature_cols = feature_cols
        self.target_col = f"target_h{horizon}"
        self.lookback = lookback

        valid = self.df.dropna(subset=feature_cols + [self.target_col])
        self.features = valid[feature_cols].to_numpy(dtype=np.float32)
        self.targets = valid[self.target_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return max(0, len(self.features) - self.lookback)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.lookback].copy()
        y = self.targets[idx + self.lookback - 1]
        return torch.from_numpy(x), torch.tensor(y)
