#!/usr/bin/env python
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine neighbors in the circle
    right_neighbor = (rank + 1) % size
    left_neighbor = (rank - 1) % size

    my_token = rank * 100
    print(f"Rank {rank}: Starting token value is {my_token}")

    # 1. Post a non-blocking receive from the left neighbor
    # This prepares the buffer so the incoming message won't get dropped or blocked
    recv_request = comm.irecv(source=left_neighbor, tag=101)

    # 2. Send our token to the right neighbor (blocking or non-blocking is fine now)
    comm.send(my_token, dest=right_neighbor, tag=101)

    # 3. Wait for our left neighbor's token to securely arrive
    incoming_token = recv_request.wait()

    print(f"Rank {rank}: Successfully passed the ring! Received token {incoming_token} from Rank {left_neighbor}")

if __name__ == "__main__":
    main()
