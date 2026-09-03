#!/usr/bin/env python
import time
import os
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    # 1. Capture Environment Variables set by torchrun/LSF
    rank       = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    master_addr = os.environ.get("MASTER_ADDR", "Not Set")
    master_port = os.environ.get("MASTER_PORT", "Not Set")

    # 2. Critical: Set the GPU before initializing NCCL
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
    
    hostname = socket.gethostname()
    # Get the IP address associated with the management network if possible
    try:
        ip_addr = socket.gethostbyname(hostname)
    except:
        ip_addr = "Unknown"

    print(f"[Rank {rank}] Host: {hostname} | IP: {ip_addr} | Local Rank: {local_rank}")
    
    if rank == 0:
        print(f"--- Distributed Config ---")
        print(f"Master Addr: {master_addr}:{master_port}")
        print(f"World Size:  {world_size}")
        print(f"Interface:   {os.environ.get('NCCL_SOCKET_IFNAME', 'DEFAULT')}")
        print(f"--------------------------\n")

    # 3. Initialize Process Group
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, hostname

def main():
    try:
        rank, local_rank, world_size, hostname = setup_distributed()
        
        # Test 1: Simple Barrier
        dist.barrier()
        if rank == 0: print("✅ Phase 1: Process Group Initialized & Barrier Passed")

        # Test 2: All-Reduce (Moving data across the network)
        # Each rank contributes its (rank + 1) to the sum
        tensor = torch.ones(1).cuda(local_rank) * (rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        expected_sum = (world_size * (world_size + 1)) / 2
        actual_sum = tensor.item()
        
        if rank == 0:
            print(f"✅ Phase 2: All-Reduce Successful. Total Sum: {actual_sum} (Expected: {expected_sum})")

        # Test 3: Dummy DDP Model
        model = torch.nn.Linear(10, 10).cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
        
        # Forward pass test
        input_data = torch.randn(20, 10).cuda(local_rank)
        output = model(input_data)
        
        if rank == 0:
            print("✅ Phase 3: DDP Model Initialized and Forward Pass complete\n")
            print("FULL DDP CHECK PASSED SUCCESSFULLY")

    except Exception as e:
        print(f"❌ ERROR on Rank {os.environ.get('RANK')}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        #current_rank = int(os.environ.get("RANK", -1))
        #if current_rank == 0:
        #    time.sleep(5)
        #else:
        #    time.sleep(1)

        #time.sleep(2)

if __name__ == "__main__":
    main()
