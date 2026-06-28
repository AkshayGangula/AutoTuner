/*
 * Hybrid CPU+GPU driver: MPI + OpenMP host work + CUDA SAXPY (1D vectors).
 *
 * CLI: --size N --iterations M (same convention as other suite CUDA drivers).
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#ifdef USE_NVTX
#include <nvToolsExt.h>
#define HYBRID_OMP_RANGE(name)                                                   \
  nvtxRangePushA(name);
#define HYBRID_OMP_RANGE_END() nvtxRangePop();
#else
#define HYBRID_OMP_RANGE(name)
#define HYBRID_OMP_RANGE_END()
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SIZE 262144
#define DEFAULT_ITERATIONS 50

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err),\
              __LINE__);                                                       \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

static int world_rank = 0;
static int world_size = 1;

__global__ static void saxpy_kernel(double a, const double *x, double *y,
                                   size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

static void print_usage(const char *prog) {
  if (world_rank != 0)
    return;
  printf("Usage: %s [OPTIONS]\n", prog);
  printf(
      "  --size N           Total double elements across all MPI ranks "
      "(partitioned; default %d)\n",
      DEFAULT_SIZE);
  printf("  --iterations M     Repeat count (default %d)\n", DEFAULT_ITERATIONS);
  printf("  -h, --help         This help\n");
}

static void hybrid_run(int total_elems, int iterations) {
  if (total_elems < 1)
    total_elems = 1;
  if (iterations < 1)
    iterations = 1;

  /* Partition global work so each rank allocates ~total/world_size on its GPU.
     Without this, N ranks sharing one GPU each cudaMalloc(--size) → N× VRAM blow-up. */
  long long te = (long long)total_elems;
  if (te < (long long)world_size)
    te = (long long)world_size;
  long long base = te / (long long)world_size;
  int rem = (int)(te % (long long)world_size);
  size_t n = (size_t)(base + (world_rank < rem ? 1LL : 0LL));
  if (n < 1)
    n = 1;
  double *h_buf = (double *)malloc(n * sizeof(double));
  if (!h_buf) {
    fprintf(stderr, "Rank %d: host malloc failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double *d_x = nullptr;
  double *d_y = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_x, n * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&d_y, n * sizeof(double)));

  HYBRID_OMP_RANGE("hybrid_omp_init");
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) {
    double t = (double)(world_rank + 1) * 0.0001 * (double)i;
    h_buf[i] = sin(t);
  }
  HYBRID_OMP_RANGE_END();

  CHECK_CUDA(
      cudaMemcpy(d_x, h_buf, n * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_y, h_buf, n * sizeof(double), cudaMemcpyHostToDevice));

  const double a = 1.00000027;
  const int threads = 256;
  dim3 block(threads);
  dim3 grid((unsigned)((n + (size_t)threads - 1) / (size_t)threads));

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  for (int it = 0; it < iterations; it++) {
    double cpu_sum = 0.0;
    HYBRID_OMP_RANGE("hybrid_omp_loop");
#pragma omp parallel for reduction(+ : cpu_sum) schedule(static)
    for (size_t i = 0; i < n; i++) {
      cpu_sum += h_buf[i] * (1.0 + 1e-15 * (double)it);
    }
    HYBRID_OMP_RANGE_END();

    saxpy_kernel<<<grid, block>>>(a, d_x, d_y, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double red_in = cpu_sum + (double)it * 1e-18;
    double red_out = 0.0;
    MPI_Allreduce(&red_in, &red_out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    (void)red_out;
  }

  double t1 = MPI_Wtime();
  double elapsed = t1 - t0;

  if (world_rank == 0) {
    /* ~3 flops per global element per iteration (SAXPY + OpenMP touch model). */
    double flops = (double)iterations * (3.0 * (double)te);
    double gflops =
        (elapsed > 1e-9) ? (flops / (elapsed * 1e9)) : 0.0;
    printf("HYBRID_VEC_GPU: Size=%d, Iterations=%d, Time=%.6f sec, "
           "Throughput=%.2f GFLOPS\n",
           total_elems, iterations, elapsed, gflops);
  }

  free(h_buf);
  cudaFree(d_x);
  cudaFree(d_y);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm node_comm;
  int local_rank = 0;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &node_comm);
  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
  }

  int num_devices = 0;
  cudaError_t err = cudaGetDeviceCount(&num_devices);
  if (err != cudaSuccess) {
    fprintf(stderr, "Rank %d: cudaGetDeviceCount failed: %s\n", world_rank,
            cudaGetErrorString(err));
    num_devices = 0;
  }
  if (num_devices == 0) {
    fprintf(stderr, "Rank %d: No CUDA devices — aborting.\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CHECK_CUDA(cudaSetDevice(local_rank % num_devices));

  if (world_rank == 0) {
    printf("=== Hybrid CPU+GPU vector (CUDA SAXPY + OpenMP + MPI) ===\n");
    printf("MPI ranks: %d | GPUs visible on this node: %d\n", world_size,
           num_devices);
  }

  int size = DEFAULT_SIZE;
  int iterations = DEFAULT_ITERATIONS;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      print_usage(argv[0]);
      MPI_Finalize();
      return 0;
    }
    if (!strcmp(argv[i], "--size") && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    } else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) {
      iterations = atoi(argv[i + 1]);
      i++;
    }
  }

  hybrid_run(size, iterations);
  MPI_Finalize();
  return 0;
}
