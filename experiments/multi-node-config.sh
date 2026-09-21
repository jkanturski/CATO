#BSUB -J train_4gpu_2nodes
#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -q gpu

torchrun \
  --nnodes=2 \
  --nproc_per_node=2 \
  --rdzv_id=scaling_4gpu \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:29500 \
  train_lstm_tcn_ddp.py --model lstm --horizon 1
