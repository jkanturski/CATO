#!/usr/bin/env python
from mpi4py import MPI
import random
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Rank 0 sets the workload and starts the timer
    if rank == 0:
        total_darts = 5_000_000  # 5 Million darts
        print(f"Rank 0: Starting Monte Carlo Pi calculation with {total_darts} darts across {size} ranks.")
        start_time = time.time()
    else:
        total_darts = None

    # 2. BROADCAST: Share the total number of darts with all ranks
    total_darts = comm.bcast(total_darts, root=0)

    # 3. CHUNK THE WORK: Divide darts equally among ranks
    local_darts = total_darts // size
    local_hits = 0

    # 4. COMPUTE: Throw the local darts
    # We use a random float between -1.0 and 1.0 for x and y
    for _ in range(local_darts):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        
        # Check if the dart landed inside the unit circle
        if (x**2 + y**2) <= 1.0:
            local_hits += 1

    print(f"Rank {rank}: Threw {local_darts} darts, got {local_hits} hits.")

    # 5. REDUCE: Combine all local hits back to Rank 0
    global_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)

    # 6. Rank 0 calculates the final value of Pi
    if rank == 0:
        # We might have lost a few darts to rounding division, so we recalculate the exact total
        actual_total_darts = local_darts * size 
        pi_estimate = 4.0 * (global_hits / actual_total_darts)
        end_time = time.time()
        
        print("\n--- RESULTS ---")
        print(f"Estimated Pi : {pi_estimate}")
        print(f"Actual Pi    : 3.1415926535...")
        print(f"Time Elapsed : {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
