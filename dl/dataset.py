import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 60):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.total_samples = len(self.features) - self.seq_len

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.features[idx : idx + self.seq_len], self.targets[idx + self.seq_len]

def prepare_dataloader(dataset, global_rank, world_size, batch_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )
    return dataloader, sampler
