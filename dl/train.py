import os
import torch
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

# Import from our local files
from dataset import CryptoTimeSeriesDataset, prepare_dataloader
from model import CryptoTCN  # or CryptoLSTM

def main():
    # 1. Initialize DDP context (torchrun sets these env vars automatically)
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    
    # 2. Setup Data
    raw_features = np.random.randn(100000, 15) # Replace with Polars data load
    raw_targets = np.random.randn(100000)
    
    dataset = CryptoTimeSeriesDataset(raw_features, raw_targets, seq_len=60)
    train_loader, sampler = prepare_dataloader(dataset, global_rank, world_size, batch_size=512)
    
    # 3. Setup Model & Optimizer
    model = CryptoTCN(input_size=15).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    # 4. Training Loop
    epochs = 50
    for epoch in range(epochs):
        sampler.set_epoch(epoch) # CRITICAL for DDP shuffling
        
        for x, y in train_loader:
            x, y = x.cuda(local_rank, non_blocking=True), y.cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast():
                output = model(x)
                loss = torch.nn.functional.mse_loss(output.squeeze(), y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        if global_rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
