#!/bin/bash
#BSUB -J crypto_lstm_xgb
#BSUB -q night
#BSUB -n 12
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o ddp_hybrid.%J.out
#BSUB -e ddp_hybrid.%J.err

export NCCL_SOCKET_IFNAME=clpriv
export NCCL_IB_DISABLE=0

# 1. Initialize Conda environment correctly
source /gpfs/catosys/opence/anaconda3/etc/profile.d/conda.sh
conda activate torch2_p9

# 2. Extract Master Node IPv4 Address
nodes=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v login)
master_node=$(head -n 1 <<< "$nodes")

# Enforce IPv4 lookup (ahostsv4) to prevent IPv6 colon parsing collisions in torchrun
master_addr=$(getent ahostsv4 $master_node | awk '{print $1}' | head -n 1)

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# 3. Scatter tasks across allocated nodes
for node in $nodes; do
    blaunch $node torchrun \
        --nnodes=6 \
        --nproc_per_node=2 \
        --rdzv_id=$LSB_JOBID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${master_addr}:29505 \
        train_hybrid.py &
done

wait
