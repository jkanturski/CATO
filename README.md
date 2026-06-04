### tryb interaktywny węzła obliczeniowego
bsub -Is -n 2 -q interactive bash


### przygotowanie środowiska
module purge
module load compilers/nvidia/hpc_sdk/21.9

### ustawienie zmiennej dla condy
source /gpfs/catosys/opence/anaconda3/etc/profile.d/conda.sh
