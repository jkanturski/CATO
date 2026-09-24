from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Error: This script requires at least 2 MPI ranks.")
        return

    MY_TAG = 42

    if rank == 0:
        payload = {
            'model_type': 'ResNet50',
            'learning_rate': 0.001,
            'epochs': 100
        }
        print(f"Rank 0: Preparing to send data: {payload}")
        comm.send(payload, dest=1, tag=MY_TAG)
        print("Rank 0: Data successfully sent!")

    elif rank == 1:
        print("Rank 1: Waiting to receive data...")
        received_data = comm.recv(source=0, tag=MY_TAG)
        print(f"Rank 1: Data successfully received!")
        print(f"Rank 1: Contents: {received_data}")

if __name__ == "__main__":
    main()
