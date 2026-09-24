#!/usr/bin/env python
from mpi4py import MPI
import random

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Every rank generates a random local maximum between 1 and 100
    local_val = random.randint(1, 100)
    print(f"Rank {rank}: My local random value is {local_val}")

    # ALLREDUCE: We want to find the absolute maximum value across the whole cluster.
    # Notice there is NO root=0 parameter here!
    global_max = comm.allreduce(local_val, op=MPI.MAX)

    # Every rank can now print the global maximum
    print(f"Rank {rank}: The highest value found anywhere on the cluster was {global_max}")

if __name__ == "__main__":
    main()
