#!/usr/bin/env python
from mpi4py import MPI
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        return

    TAG_DATA = 99

    if rank == 0:
        print("Rank 0: Posted non-blocking receive. Waiting for data from Rank 1...")
        request = comm.irecv(source=1, tag=TAG_DATA)
        
        print("Rank 0: Doing heavy math while waiting for the network...")
        for i in range(3):
            print(f"Rank 0: Computing step {i+1}/3...")
            time.sleep(0.5) 
            
        print("Rank 0: Math done. Now waiting explicitly for the message to land...")
        data = request.wait()
        print(f"Rank 0: Success! Received data: {data}")

    elif rank == 1:
        payload = [10, 20, 30, 40, 50]
        time.sleep(1.0) 
        print(f"Rank 1: Sending data payload over the network...")
        comm.send(payload, dest=0, tag=TAG_DATA)

if __name__ == "__main__":
    main()
