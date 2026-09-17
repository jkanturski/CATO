#!/bin/bash
#BSUB -J "poprad_cds_deep"
#BSUB -q night
#BSUB -n 12
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=exclusive_process:j_exclusive=yes"
#BSUB -o poprad_cds.%J.log
#BSUB -e poprad_cds.%J.err
#
# Launches distributed training of the LSTM/TCN/hybrid forecasters
# (Sections III.C/D/E, V of paper.tex) on the CATO cluster.
#
# Usage:
#   bsub < submit_deep_models.sh
#
# Edit MODEL/HORIZON/SCRIPT below to select which of the three deep
# configurations and which forecasting horizon to run. For the full
# scaling experiment (Table II of paper.tex), resubmit with different
# #BSUB -n / -R / -gpu values (1 GPU/1 node, 2 GPU/1 node, 4 GPU/2 nodes,
# 12 GPU/6 nodes) and record wall-clock time from the LSF job log.

MODEL=tcn          # one of: lstm, tcn (used only when SCRIPT=train_lstm_tcn_ddp.py)
HORIZON=7          # one of: 1, 7, 30
SCRIPT=train_lstm_tcn_ddp.py   # train_lstm_tcn_ddp.py (lstm/tcn) or train_hybrid.py
EXTRA_ARGS="--model $MODEL --horizon $HORIZON"   # for train_hybrid.py, use "--horizon $HORIZON" only

module purge
module load compilers/nvidia/hpc_sdk/nompi/21.9

# Master address = first host in allocation
export MASTER_ADDR=$(echo $LSB_MCPU_HOSTS | awk '{print $1}')
export MASTER_PORT=29505
unset MASTER_ADDR  # LSF injects an incompatible value; resolve IPv4 below instead

NODES=($(echo $LSB_MCPU_HOSTS | awk '{for(i=1;i<=NF;i+=2) print $i}'))
NNODES=${#NODES[@]}
MASTER_HOST=${NODES[0]}
MASTER_ADDR=$(getent ahostsv4 $MASTER_HOST | awk '{print $1}' | head -n 1)

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NNODES=$NNODES"
echo "NODES=${NODES[@]}"
echo "MODEL=$MODEL HORIZON=$HORIZON"

for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    blaunch $NODE "
        source /gpfs/catosys/opence/anaconda3/etc/profile.d/conda.sh && \
        conda activate torch2_p9 && \
        export NCCL_DEBUG=INFO && \
        export NCCL_IB_DISABLE=0 && \
        export NCCL_SOCKET_IFNAME=clpriv && \
        export NCCL_ASYNC_ERROR_HANDLING=1 && \
        export CUDA_VISIBLE_DEVICES=0,1 && \
        export KMP_DUPLICATE_LIB_OK=TRUE && \
        cd $LS_SUBCWD && \
        echo \"Starting node_rank=$i on $NODE\" && \
        torchrun \
            --nproc_per_node=2 \
            --nnodes=$NNODES \
            --node_rank=$i \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            $SCRIPT $EXTRA_ARGS
    " &
done

wait

echo "Job completed."
