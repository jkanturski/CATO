import torch
import polars as pl
import numpy as np
from torch.utils.data import Dataset

class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, parquet_path, seq_length=128):
        # Load historical Solana/Aave market data
        self.df = pl.read_parquet(parquet_path)
        self.seq_length = seq_length
        
        # Assuming 'target_return' is the label and the rest are features (e.g., L2 depth, RSI)
        self.targets = self.df["target_return"].to_numpy(dtype=np.float32)
        self.features = self.df.drop(["target_return", "timestamp"]).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df) - self.seq_length

    def __getitem__(self, idx):
        # Extract sliding window
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x), torch.tensor(y)
