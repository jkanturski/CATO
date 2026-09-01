#!/bin/bash
#BSUB -J crypto_lstm_xgb
#BSUB -q night
#BSUB -n 12
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -o ddp_hybrid.%J.out
#BSUB -e ddp_hybrid.%J.err

export NCCL_SOCKET_IFNAME=clpriv
export NCCL_IB_DISABLE=0

# Extract master node IP
nodes=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v login)
master_node=$(head -n 1 <<< "$nodes")
master_addr=$(ssh $master_node "hostname -i")

torchrun \
    --nnodes=6 \
    --nproc_per_node=2 \
    --rdzv_id=$LSB_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:29505 \
    train_hybrid.py
