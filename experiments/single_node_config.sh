#BSUB -J train_2gpu_1node
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -q short
#BSUB -o train_2gpu_1node.%J.log
#BSUB -e train_2gpu_1node.%J.err

# Added --model flag and updated script name
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_lstm_tcn_ddp.py --model lstm --horizon 1
