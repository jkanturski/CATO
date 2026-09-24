#!/usr/bin/env python
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Rank 0 creates the initial data
    if rank == 0:
        # Create a list with one item for each rank
        # e.g., if size is 4, dataset = [0, 10, 20, 30]
        dataset = [i * 10 for i in range(size)]
        print(f"Rank 0: Starting with dataset: {dataset}")
    else:
        # Workers must declare the variable as None before the collective call
        dataset = None

    # 2. SCATTER: Rank 0 chops the dataset and deals one piece to each rank
    # Notice that EVERY rank executes this line!
    local_data = comm.scatter(dataset, root=0)
    print(f"Rank {rank}: Received data '{local_data}' from root.")

    # 3. COMPUTE: Every rank modifies its own local piece
    local_result = local_data + 5
    print(f"Rank {rank}: Computed result '{local_result}'.")

    # 4. GATHER: Rank 0 collects all the local_results back into a new list
    final_dataset = comm.gather(local_result, root=0)

    if rank == 0:
        print(f"Rank 0: Gathered final dataset: {final_dataset}")

if __name__ == "__main__":
    main()
