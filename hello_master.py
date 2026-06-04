from mpi4py import MPI
import socket

def main():
    # Setup the MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    # The Master/Worker split
    if rank == 0:
        print(f"[{hostname}] MASTER (Rank 0): Welcome everyone! We have a total of {size} processes running today.")
    else:
        print(f"[{hostname}] WORKER (Rank {rank}): Reporting for duty!")

if __name__ == "__main__":
    main()
