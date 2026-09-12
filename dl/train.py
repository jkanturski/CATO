import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from dataset import CryptoTimeSeriesDataset
from model import CryptoTCN

def train():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    script_dir = Path(__file__).resolve().parent

    full_dataset = CryptoTimeSeriesDataset("/home/jkanturski/uksw/CATO/dl/data/train_solana_aave.parquet")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Enforce identical random split across all 12 DDP ranks via a fixed generator seed
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=512, sampler=train_sampler, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=512, sampler=val_sampler, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    model = CryptoTCN(input_size=6, num_channels=[64, 128, 256], kernel_size=3).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    epochs = 50
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch) 
        
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.cuda(local_rank, non_blocking=True), y.cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(x)
                loss = criterion(outputs.squeeze(), y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(local_rank, non_blocking=True), y.cuda(local_rank, non_blocking=True)
                with autocast():
                    outputs = model(x)
                    loss = criterion(outputs.squeeze(), y)
                val_loss += loss.item()
        
        # Sync metrics across all GPUs
        metrics = torch.tensor([train_loss / len(train_loader), val_loss / len(val_loader)]).cuda(local_rank)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        
        # Rank 0 Logging & Checkpointing
        if global_rank == 0:
            print(f"Epoch {epoch:02d} | Train Loss: {metrics[0]:.6f} | Val Loss: {metrics[1]:.6f}")
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': metrics[1].item()
                }
                
                checkpoint_dir = script_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save(checkpoint, str(checkpoint_dir / f"tcn_epoch_{epoch}.pt"))

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
