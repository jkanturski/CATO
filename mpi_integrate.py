#!/usr/bin/env python
from mpi4py import MPI

def f(x):
    """The function we are integrating: f(x) = x^2"""
    return x ** 2

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Boundaries of integration
    a = 0.0
    b = 100.0

    if rank == 0:
        total_steps = 1000  # Number of rectangles
    else:
        total_steps = None

    # 1. BROADCAST: Rank 0 shares the total_steps with all workers
    total_steps = comm.bcast(total_steps, root=0)

    # Calculate width of a single rectangle
    dx = (b - a) / total_steps

    # 2. CHUNK THE WORK: How many steps does each rank compute?
    # (Assuming total_steps is perfectly divisible by size for simplicity)
    local_steps = total_steps // size
    
    # Calculate local start and end bounds based on rank ID
    local_a = a + (rank * local_steps * dx)
    
    # 3. COMPUTE: Calculate the area for this rank's specific slice
    local_area = 0.0
    for i in range(local_steps):
        x = local_a + (i * dx)
        local_area += f(x) * dx
        
    print(f"Rank {rank}: Calculated partial area = {local_area:.4f}")

    # 4. REDUCE: Sum all local areas into a single total_area on Rank 0
    total_area = comm.reduce(local_area, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"\nRank 0: Total Integrated Area = {total_area:.4f}")
        # Mathematical exact answer is (100^3)/3 = 333333.3333

if __name__ == "__main__":
    main()
