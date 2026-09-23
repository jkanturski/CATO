#BSUB -J train_1gpu_1node
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q short
#BSUB -o train_1gpu_1node.%J.log
#BSUB -e train_1gpu_1node.%J.err

torchrun --standalone --nnodes=1 --nproc_per_node=1 train_lstm_tcn_ddp.py --model lstm --horizon 1