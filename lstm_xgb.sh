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

# 1. Initialize Conda environment correctly
source /gpfs/catosys/opence/anaconda3/etc/profile.d/conda.sh
conda activate torch2_p9

export NCCL_SOCKET_IFNAME=clpriv
export NCCL_IB_DISABLE=0

# 2. Extract Master Node IP WITHOUT SSH
nodes=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v login)
master_node=$(head -n 1 <<< "$nodes")
# getent resolves the IP locally without requiring SSH authentication
master_addr=$(getent hosts $master_node | awk '{print $1}')

# 3. Scatter the task across all allocated nodes using LSF blaunch
for node in $nodes; do
    blaunch $node torchrun \
        --nnodes=6 \
        --nproc_per_node=2 \
        --rdzv_id=$LSB_JOBID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${master_addr}:29505 \
        train_hybrid.py &
done

# Wait for all background blaunch tasks to complete
wait
