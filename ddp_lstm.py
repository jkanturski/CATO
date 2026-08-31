#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def main():
    # Initialize the process group with the NCCL backend
    dist.init_process_group(backend="nccl")
    
    # torchrun automatically injects these environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    input_size, hidden_size, num_layers, num_classes = 10, 32, 2, 2
    batch_size, seq_length, epochs = 64, 20, 10
    
    # Instantiate model and pin it to the specific local GPU
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate dummy sequential dataset (e.g., 1024 samples)
    dataset = TensorDataset(
        torch.randn(1024, seq_length, input_size),
        torch.randint(0, num_classes, (1024,))
    )
    
    # DistributedSampler ensures each GPU gets a unique subset of the data
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    if global_rank == 0:
        print("Starting Distributed LSTM Training on CATO...")

    for epoch in range(epochs):
        # Crucial: Set the epoch on the sampler to shuffle data correctly across ranks
        sampler.set_epoch(epoch)
        model.train()
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(local_rank)
            y_batch = y_batch.to(local_rank)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Only print from the master rank to prevent terminal spam
        if global_rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
