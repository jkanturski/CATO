from mpi4py import MPI
import socket

def main():
    # 1. The Communicator: The network connecting all our processes
    comm = MPI.COMM_WORLD
    
    # 2. Size: How many total processes are running?
    size = comm.Get_size()
    
    # 3. Rank: What is MY specific ID number?
    rank = comm.Get_rank()
    
    # 4. Hostname: What physical computer am I running on?
    hostname = socket.gethostname()
    
    # 5. The Output
    print(f"Hello World! I am Rank {rank} out of {size} running on {hostname}.")

if __name__ == "__main__":
    main()
