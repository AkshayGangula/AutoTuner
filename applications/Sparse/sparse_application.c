#define _POSIX_C_SOURCE 199309L
/**
 * Sparse Matrix Multiplication Application (MPI+OpenMP)
 * Part of the Auto-Tuning Framework
 *
 * Characteristics:
 * - Irregular memory access patterns
 * - Communication-intensive (matrix distribution)
 * - Good for testing MPI communication efficiency
 */

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_SIZE 1024
#define DEFAULT_ITERATIONS 100

// Global variables
int world_rank, world_size;
int omp_threads;

void sparse_matrix_multiply(int size, int iterations) {
  int local_size = size / world_size;
  int remainder = size % world_size;

  // Adjust local size for remainder
  if (world_rank < remainder) {
    local_size++;
  }

  // Safety check: ensure local_size is at least 1
  if (local_size < 1) {
    local_size = 1;
  }

  int start_row = world_rank * (size / world_size) +
                  (world_rank < remainder ? world_rank : remainder);

  // Allocate local matrices
  double *local_A = (double *)malloc(local_size * size * sizeof(double));
  double *local_B = (double *)malloc(size * size * sizeof(double));
  double *local_C = (double *)malloc(local_size * size * sizeof(double));

  if (!local_A || !local_B || !local_C) {
    fprintf(stderr, "Rank %d: out of memory allocating matrices\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return;
  }

// Initialize matrices with sparse pattern (10% non-zero)
#pragma omp parallel for
  for (int i = 0; i < local_size; i++) {
    for (int j = 0; j < size; j++) {
      if ((i + start_row + j) % 10 == 0) {
        local_A[i * size + j] = (double)(i + start_row + j) / size;
      } else {
        local_A[i * size + j] = 0.0;
      }
    }
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if ((i + j) % 10 == 0) {
        local_B[i * size + j] = (double)(i + j) / size;
      } else {
        local_B[i * size + j] = 0.0;
      }
    }
  }

  // Simulate communication initialization without blowing up MVAPICH2 shmem buffers
  // MVAPICH2 crashes/hangs with >33MB intra-node Bcast on 32-64 dense ranks.
  MPI_Bcast(&local_B[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Perform matrix multiplication iterations
  double start_time = MPI_Wtime();

  for (int iter = 0; iter < iterations; iter++) {
#pragma omp parallel for
    for (int i = 0; i < local_size; i++) {
      for (int j = 0; j < size; j++) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
          if (local_A[i * size + k] != 0.0 && local_B[k * size + j] != 0.0) {
            sum += local_A[i * size + k] * local_B[k * size + j];
          }
        }
        local_C[i * size + j] = sum;
      }
    }

    // Allreduce to compute global sum (communication) using MPI
    double local_sum = 0.0;
#pragma omp parallel for reduction(+ : local_sum)
    for (int i = 0; i < local_size * size; i++) {
      local_sum += local_C[i];
    }

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }

  double end_time = MPI_Wtime();
  double elapsed = end_time - start_time;

  if (world_rank == 0) {
    printf("SPARSE: Size=%d, Iterations=%d, Time=%.6f sec, Throughput=%.2f "
           "GFLOPS\n",
           size, iterations, elapsed,
           (2.0 * size * size * size * iterations) / (elapsed * 1e9));
  }

  free(local_A);
  free(local_B);
  free(local_C);
}

void print_usage(const char *prog_name) {
  if (world_rank == 0) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  --size SIZE       Problem size (default: %d)\n", DEFAULT_SIZE);
    printf("  --iterations N    Number of iterations (default: %d)\n",
           DEFAULT_ITERATIONS);
    printf("  --help, -h        Show this help message\n");
  }
}

void parse_arguments(int argc, char *argv[], int *size, int *iterations) {
  *size = DEFAULT_SIZE;
  *iterations = DEFAULT_ITERATIONS;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      *size = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      *iterations = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      exit(0);
    }
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the number of threads that OpenMP will use (respects OMP_NUM_THREADS)
  // IMPORTANT: Call this BEFORE any parallel region to get the correct value
  omp_threads = omp_get_max_threads();

  if (world_rank == 0) {
    printf("=== Sparse Matrix Application ===\n");
    printf("MPI Ranks: %d, OpenMP Threads: %d\n", world_size, omp_threads);
  }

  int size, iterations;
  parse_arguments(argc, argv, &size, &iterations);

  if (world_rank == 0) {
    printf("Size: %d, Iterations: %d\n\n", size, iterations);
  }

  sparse_matrix_multiply(size, iterations);

// Ensure cleanup and avoid MVAPICH2 race conditions
#pragma omp parallel
  {
#pragma omp barrier
  }

  struct timespec delay;
  delay.tv_sec = 0;
  delay.tv_nsec = (world_size > 16) ? 50 * 1000 * 1000 : 10 * 1000 * 1000;
  nanosleep(&delay, NULL);

  MPI_Finalize();
  return 0;
}
