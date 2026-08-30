#!/bin/bash
#BSUB -J pi_monte_carlo        # The name of the job in the queue
#BSUB -n 4                     # Request exactly 4 cores/slots
#BSUB -W 00:10                 # Maximum wall-clock time requested (10 minutes)
#BSUB -o pi_output_%J.txt      # Where to save the standard output (%J becomes the Job ID)
#BSUB -e pi_error_%J.txt       # Where to save the error output

# 1. Prepare the module environment
module purge
module load compilers/nvidia/hpc_sdk/21.9

# 2. Initialize Conda for a non-interactive shell
# (This hook is required because batch scripts run in the background without a normal terminal)
eval "$(conda shell.bash hook)"
conda activate torch2_p9

# 3. Execute the code
echo "Starting MPI Job across 4 ranks..."
mpiexec -n 4 python pi_solution.py
echo "Job complete!"
