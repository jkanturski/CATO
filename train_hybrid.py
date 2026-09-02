import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import xgboost as xgb

class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        final_hidden = out[:, -1, :] 
        prediction = self.fc(final_hidden)
        return prediction, final_hidden

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_pipeline():
    local_rank = setup()
    
    model = CryptoLSTM(input_size=15, hidden_size=64, num_layers=2).cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # --- Phase 2: Feature Extraction and Synchronization ---
    ddp_model.eval()
    
    with torch.no_grad():
        # [!] FIXED: Generate dummy tensors so the script actually compiles.
        # Replace this with your actual DataLoader output later.
        batch_size = 32
        inputs = torch.randn(batch_size, 10, 15).cuda(local_rank)
        targets = torch.randn(batch_size, 1).cuda(local_rank)
        
        # [!] FIXED: Uncommented the forward pass so final_hidden is defined.
        prediction, final_hidden = ddp_model(inputs)
        
        local_features = final_hidden.contiguous()
        local_targets = targets.contiguous() 
    
        world_size = dist.get_world_size()
        
        gathered_features_list = [torch.zeros_like(local_features) for _ in range(world_size)]
        gathered_targets_list = [torch.zeros_like(local_targets) for _ in range(world_size)]
    
        dist.all_gather(gathered_features_list, local_features)
        dist.all_gather(gathered_targets_list, local_targets)

    # [!] FIXED: Explicitly release the PyTorch NCCL process group 
    # to free the GPU for XGBoost in LSF exclusive_process mode.
    dist.destroy_process_group()
    torch.cuda.empty_cache()

    # --- Phase 3: GPU-Accelerated XGBoost (Global Rank 0 Only) ---
    # Extract the global rank explicitly from torchrun's environment variables
    global_rank = int(os.environ["RANK"])
    
    if global_rank == 0:
        gathered_features = torch.cat(gathered_features_list, dim=0).cpu().numpy()
        gathered_targets = torch.cat(gathered_targets_list, dim=0).cpu().numpy()
        
        dtrain = xgb.DMatrix(gathered_features, label=gathered_targets)
        
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist', # Power9/Older XGBoost GPU syntax
            'learning_rate': 0.05,
            'max_depth': 6
        }
        
        bst = xgb.train(params, dtrain, num_boost_round=100)
        bst.save_model("hybrid_crypto_model.json")
        print(f"XGBoost training complete. Model saved by Global Rank {global_rank}.")
    
if __name__ == "__main__":
    train_pipeline()
