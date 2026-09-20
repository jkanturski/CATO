"""
train_hybrid.py — DDP-trained TCN feature extractor + XGBoost regressor
for Poland CDS 5Y forecasting (Section III.E and Fig. 1 of paper.tex).

Phase 1: train the TCN backbone under PyTorch DDP (mirrors the earlier
prototype /Users/rklopotek/CATO/train_hybrid.py, extended from synthetic
data to the real CDS dataset and from a single forward pass to full
training), with early stopping on validation RMSE (mirrors
train_lstm_tcn_ddp.py) to avoid overfitting the TCN backbone before
Phase 2.
Phase 2: extract penultimate-layer embeddings, all_gather them across
ranks, and train an XGBoost regressor on rank 0 using the gathered
embeddings plus a small set of raw lag-1 features.

Usage (single GPU/CPU, local run):
    python train_hybrid.py --horizon 7

Usage (multi-node on CATO, via torchrun/LSF): see submit_deep_models.sh.
"""
import argparse
import os

# NOTE: on some macOS/conda environments, importing torch before xgboost
# triggers a libomp (OpenMP) symbol conflict that crashes the process
# (SIGABRT/segfault) once xgboost's C++ core is invoked. Importing
# xgboost first (even though it is only used in main()) avoids this. This
# is a local-environment quirk observed during development and is not
# expected to reproduce on the CATO cluster's Linux/conda (torch2_p9)
# environment, but the safe import order is kept here regardless.
import xgboost as xgb  # noqa: F401  (imported early to avoid OpenMP conflict)

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import CDSSequenceDataset
from metrics import all_metrics, rmse
from models import TCNForecaster

RAW_LAG1_EXTRA = [
    "POLAND CDS USD SR 2Y Corp_lag1",
    "POLAND CDS USD SR 10Y Corp_lag1",
    "EPUCGLCP Index_lag1",
    "GPRXGPRD Index_lag1",
]


def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ



def train_tcn_phase(args, raw_feature_cols, device, distributed, local_rank, global_rank):
    train_ds = CDSSequenceDataset(os.path.join(args.data_dir, "train.parquet"),
                                   raw_feature_cols, args.horizon, args.lookback)
    val_ds = CDSSequenceDataset(os.path.join(args.data_dir, "val.parquet"),
                                 raw_feature_cols, args.horizon, args.lookback)

    sampler = DistributedSampler(train_ds) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=sampler, shuffle=(sampler is None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TCNForecaster(len(raw_feature_cols), num_channels=(64, 64, 64),
                           num_horizons=1).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience, patience_ctr = 10, 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).squeeze(-1).cpu().numpy()
                preds.append(pred)
                trues.append(y.numpy())
        val_rmse = rmse(np.concatenate(trues), np.concatenate(preds))

        if global_rank == 0 and (epoch % 10 == 0 or epoch == args.epochs - 1):
            print(f"[hybrid-tcn h={args.horizon}] epoch {epoch}: val RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_ctr = 0
            backbone = model.module if distributed else model
            best_state = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if global_rank == 0:
                    print(f"[hybrid-tcn h={args.horizon}] early stopping at epoch {epoch} "
                          f"(best val RMSE={best_val_rmse:.4f})")
                break

    # Restore the best-validation-RMSE weights before extracting embeddings,
    # so Phase 2 (XGBoost) trains on features from the best TCN checkpoint
    # rather than a possibly overfit final epoch.
    if best_state is not None:
        backbone = model.module if distributed else model
        backbone.load_state_dict(best_state)

    return model

def extract_embeddings(model, dataset, device, batch_size=256):
    """Run model in eval mode on Rank 0 without DDP duplication."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    backbone = model.module if isinstance(model, DDP) else model
    backbone.eval()

    embeds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, emb = backbone(x, return_embedding=True)
            embeds.append(emb.cpu().numpy())
            targets.append(y.numpy())

    return np.concatenate(embeds, axis=0), np.concatenate(targets, axis=0)


def run_horizon(args):
    """Run the full hybrid pipeline for one horizon and return a metrics
    dict (None on non-zero ranks in distributed mode, where only rank 0
    computes final metrics)."""
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

    sample = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    feature_cols = [c for c in sample.columns if "_lag" in c]
    raw_feature_cols = [c for c in feature_cols if c.endswith("_lag1")]

    model = train_tcn_phase(args, raw_feature_cols, device, distributed,
                             local_rank, global_rank)

    train_ds = CDSSequenceDataset(os.path.join(args.data_dir, "train.parquet"),
                                   raw_feature_cols, args.horizon, args.lookback)
    test_ds = CDSSequenceDataset(os.path.join(args.data_dir, "test.parquet"),
                                  raw_feature_cols, args.horizon, args.lookback)

    if distributed:
        dist.barrier()  # Synchronize all ranks after Phase 1 TCN training

    result = None
    if global_rank == 0:
        # Rank 0 extracts embeddings locally (no 6x duplication)
        train_emb, train_y = extract_embeddings(model, train_ds, device)
        test_emb, test_y = extract_embeddings(model, test_ds, device)

        extra_cols = [c for c in RAW_LAG1_EXTRA if c in sample.columns]
        train_extra = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))[extra_cols]
        test_extra_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
        test_extra = test_extra_df[extra_cols]

        n_train, n_test = len(train_y), len(test_y)
        off = args.lookback - 1
        train_extra_arr = train_extra.to_numpy()[off: off + n_train]
        test_extra_arr = test_extra.to_numpy()[off: off + n_test]

        assert train_emb.shape[0] == train_extra_arr.shape[0], (
            f"Row mismatch: train_emb has {train_emb.shape[0]} rows, "
            f"while train_extra_arr has {train_extra_arr.shape[0]} rows."
        )

        X_train = np.concatenate([train_emb, train_extra_arr], axis=1)
        X_test = np.concatenate([test_emb, test_extra_arr], axis=1)

        reg = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                max_depth=5, learning_rate=0.05, n_estimators=300)
        reg.fit(X_train, train_y)
        test_pred = reg.predict(X_test)

        y_prev = test_extra_df["POLAND CDS USD SR 5Y Corp"].to_numpy()[off: off + n_test]
        result = all_metrics(y_prev, test_y, test_pred)
        print(f"[hybrid h={args.horizon}] test metrics: {result}")

    if distributed:
        dist.barrier()  # Keep non-zero ranks alive until Rank 0 completes Phase 2
        dist.destroy_process_group()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, choices=[1, 7, 30])
    parser.add_argument("--data-dir",
                         default="/home/jkanturski/CATO/experiments/data")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run_horizon(args)


if __name__ == "__main__":
    main()
