### tryb interaktywny węzła
```
bsub -Is -n 2 -q interactive bash
```
### przygotowanie środowiska
```
module purge  
module load compilers/nvidia/hpc_sdk/21.9
```
### sprawdzenie mpi
```
which mpiexec
```
### conda
```
source /gpfs/catosys/opence/anaconda3/etc/profile.d/conda.sh  
conda create -n nazwa_srodowiska python=3.9 -y
conda activate nazwa_srodowiska  
conda deactivate
```
### przykład uruchomienia programu
```
mpiexec -n 2 python send_dict.py
```
### instalacja mpi4py i weryfikacja
```
conda install -c conda-forge mpi4py -y  
python -c "import mpi4py; print('mpi4py version:', mpi4py.__version__)"
```
