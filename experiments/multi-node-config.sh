#BSUB -J train_4gpu_2nodes
#BSUB -q short
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -R "select[hname!='ln01' && hname!='ln02']"
#BSUB -gpu "num=2:mode=exclusive_process:j_exclusive=yes"
#BSUB -o train_4gpu_2nodes.%J.log
#BSUB -e train_4gpu_2nodes.%J.err

module purge
module load compilers/nvidia/hpc_sdk/nompi/21.9

# Master address = first host in allocation
export MASTER_ADDR=$(echo $LSB_MCPU_HOSTS | awk '{print $1}')
export MASTER_PORT=29505

# Extract node list (skipping task counts)
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
        cd $LS_SUBCWD && \
        echo \"Starting node_rank=$i on $NODE\" && \
        torchrun \
            --nproc_per_node=2 \
            --nnodes=$NNODES \
            --node_rank=$i \
            --rdzv_id=$LSB_JOBID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            train_lstm_tcn_ddp.py --model lstm --horizon 1
    " &
done

wait

echo "Job completed."