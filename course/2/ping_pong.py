from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        if rank == 0:
            print("Error: This script requires at least 2 MPI ranks.")
        return

    TAG_PONG = 11

    if rank == 0:
        ball = 100
        print(f"Rank 0: Starting Ping-Pong. Initial ball value = {ball}")
        
        # Send to Rank 1
        comm.send(ball, dest=1, tag=TAG_PONG)
        
        # Wait for the reply from Rank 1
        final_ball = comm.recv(source=1, tag=TAG_PONG)
        print(f"Rank 0: Received the ball back! Final value = {final_ball}")

    elif rank == 1:
        # Receive from Rank 0
        received_ball = comm.recv(source=0, tag=TAG_PONG)
        print(f"Rank 1: Caught the ball. Value = {received_ball}")
        
        # Modify the data
        received_ball += 50
        print(f"Rank 1: Modified ball value to {received_ball}. Sending back...")
        
        # Send back to Rank 0
        comm.send(received_ball, dest=0, tag=TAG_PONG)

if __name__ == "__main__":
    main()
