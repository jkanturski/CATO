"""
train_lstm_tcn_ddp.py — distributed (PyTorch DDP) training of the LSTM
and TCN forecasters for Poland CDS 5Y forecasting (Sections III.C/D and
V of paper.tex). Designed to be launched via torchrun on the CATO cluster
(see submit_deep_models.sh in this folder).

Usage (single GPU, local test):
    python train_lstm_tcn_ddp.py --model tcn --horizon 7

Usage (multi-node on CATO, via torchrun/LSF): see submit_deep_models.sh.
"""
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import CDSSequenceDataset
from metrics import all_metrics, rmse
from models import LSTMForecaster, TCNForecaster

def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).squeeze(-1).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)



def run_horizon(args):
    distributed = is_distributed()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, global_rank = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd
    sample = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    feature_cols = [c for c in sample.columns if "_lag" in c]
    # Restrict to raw (lag-1 only) features for the sequence models, since
    # the LSTM/TCN learn temporal structure internally (Section IV.C).
    raw_feature_cols = [c for c in feature_cols if c.endswith("_lag1")]

    train_ds = CDSSequenceDataset(os.path.join(args.data_dir, "train.parquet"),
                                   raw_feature_cols, args.horizon, args.lookback)
    val_ds = CDSSequenceDataset(os.path.join(args.data_dir, "val.parquet"),
                                 raw_feature_cols, args.horizon, args.lookback)
    test_ds = CDSSequenceDataset(os.path.join(args.data_dir, "test.parquet"),
                                  raw_feature_cols, args.horizon, args.lookback)

    if distributed:
        train_sampler = DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=train_sampler,
                               shuffle=(train_sampler is None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_size = len(raw_feature_cols)
    if args.model == "lstm":
        model = LSTMForecaster(input_size, hidden_size=64, num_layers=2, num_horizons=1)
    else:
        model = TCNForecaster(input_size, num_channels=(64, 64, 64), num_horizons=1)
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience, patience_ctr = 10, 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        trues, preds = evaluate(model, val_loader, device)
        val_rmse = rmse(trues, preds)

        if global_rank == 0:
            print(f"[{args.model} h={args.horizon}] epoch {epoch}: val RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_ctr = 0
            backbone = model.module if distributed else model
            best_state = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
            if global_rank == 0:
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save(best_state, os.path.join(
                    args.out_dir, f"{args.model}_h{args.horizon}_best.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if global_rank == 0:
                    print(f"[{args.model} h={args.horizon}] early stopping at epoch {epoch} "
                          f"(best val RMSE={best_val_rmse:.4f})")
                break

    result = None
    if global_rank == 0 and best_state is not None:
        backbone = model.module if distributed else model
        backbone.load_state_dict(best_state)
        test_trues, test_preds = evaluate(model, test_loader, device)

        # y_prev for directional accuracy: raw CDS 5Y value at the last
        # lookback timestamp for each test window.
        test_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
        valid = test_df.dropna(subset=raw_feature_cols + [f"target_h{args.horizon}"])
        y_prev = valid["POLAND CDS USD SR 5Y Corp"].to_numpy()[
            args.lookback - 1: args.lookback - 1 + len(test_trues)]

        result = all_metrics(y_prev, test_trues, test_preds)
        print(f"[{args.model} h={args.horizon}] test metrics: {result}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "tcn"], required=True)
    parser.add_argument("--horizon", type=int, required=True, choices=[1, 7, 30])
    parser.add_argument("--data-dir",
                         default="/Users/rklopotek/CATO/Poprad_2026/experiments/data")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir",
                         default="/Users/rklopotek/CATO/Poprad_2026/experiments/checkpoints")
    args = parser.parse_args()
    run_horizon(args)


if __name__ == "__main__":
    main()

