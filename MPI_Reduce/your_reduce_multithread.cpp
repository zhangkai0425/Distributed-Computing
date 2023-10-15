#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

void YOUR_Reduce_Multithread(const int *sendbuf, int *recvbuf, int count, int num_threads)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    omp_set_num_threads(num_threads);
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
            #pragma omp parallel for
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
}
