#BSUB -J "lstm_hpc"
#BSUB -q night
#BSUB -n 12
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=exclusive_process:j_exclusive=yes"
#BSUB -m "cn01 cn02 cn03 cn04 cn05 cn06"
#BSUB -o ddp_out.%J.log
#BSUB -e ddp_err.%J.log

module purge
module load compilers/nvidia/hpc_sdk/nompi/21.9

# Master address = first host in allocation
export MASTER_ADDR=$(echo $LSB_MCPU_HOSTS | awk '{print $1}')
export MASTER_PORT=29505

# Extract node list
NODES=($(echo $LSB_MCPU_HOSTS | awk '{for(i=1;i<=NF;i+=2) print $i}'))
NNODES=${#NODES[@]}

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NNODES=$NNODES"
echo "NODES=${NODES[@]}"

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
        cd $LS_SUBCWD && \
        echo \"Starting node_rank=$i on $NODE\" && \
        torchrun \
            --nproc_per_node=2 \
            --nnodes=$NNODES \
            --node_rank=$i \
            --rdzv_backend=static \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            ddp_lstm.py
    " &
done

wait

echo "Job completed."
