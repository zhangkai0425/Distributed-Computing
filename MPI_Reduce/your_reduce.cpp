#include "your_reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
// You may add your functions and variables here

void YOUR_Reduce(const int *sendbuf, int *recvbuf, int count)
{
    /*
    Modify the code here.
    Your implementation should have the same result as this MPI_Reduce
    call. However, you MUST NOT use MPI_Reduce (or like) for your hand-in
    version. Instead, you should use MPI_Send and MPI_Recv (or like). See
    the homework instructions for more information.
    */
    int rank, size;
    // Get the current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the size of the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    memcpy(recvbuf, sendbuf, count * sizeof(int));
    if (size == 1)
        return;
    int max_depth = 0;
    int temp_size = size;
    while (temp_size > 1)
    {
        max_depth++;
        temp_size = (temp_size + 1) / 2;
    }
    int step = 1;
    for (int depth = 0; depth < max_depth; depth++)
    {
        // Calculate the partner process rank
        int partner = rank ^ step;
        if (rank % (2 * step) == 0 && partner < size)
        {
            int *received = (int *)malloc(count * sizeof(int));
            MPI_Request recv_request;
            MPI_Irecv(received, count, MPI_INT, partner, 0, MPI_COMM_WORLD, &recv_request);
            // Wait for the receive to complete
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            // Apply the reduction operation (in this case, MPI_SUM)
            for (int i = 0; i < count; i++)
                recvbuf[i] += received[i];
            free(received);
        }
        else if (rank % (2 * step) == step)
        {
            int dest = rank - step;
            MPI_Request send_request;
            MPI_Isend(recvbuf, count, MPI_INT, dest, 0, MPI_COMM_WORLD, &send_request);
            // Wait for the send to complete
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        }

        step *= 2;
    }
    /*
    You may assume:
    - Data type is always `int` (MPI_INT).
    - Operation is always MPI_SUM.
    - Process to hold final results is always process 0.
    - Number of processes is 2, 4, or 8.
    - Number of elements (`count`) is 1, 16, 256, 4096, 65536, 1048576,
      16777216, or 268435456.
    For other cases, your code is allowed to produce wrong results or even
    crash. It is totally fine.
    */
}