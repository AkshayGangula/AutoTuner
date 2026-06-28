# Makefile for Unified Application Suite
# Compiles the unified applications for MPI+OpenMP auto-tuning.
# Sparse sources: Sparse/sparse_application.c → ./sparse_application
# HybridVec GPU: HybridVec/hybrid_vec_gpu.cu → ./hybrid_vec_gpu (MPI + OpenMP + CUDA SAXPY)
#
# Usage:
#   make              - GCC + OpenMP (default)
#   make COMPILER=intel  - Intel compiler + Intel MPI (for Nsight OpenMP instrumentation)

COMPILER ?= gcc

ifeq ($(COMPILER),intel)
  # Intel: use Intel MPI C wrapper; -qopenmp for OpenMP (Nsight can instrument this)
  # Workaround for Intel 19 + glibc 2.27+: pretend to be GCC 7+ so glibc skips
  # the fallback "typedef float _Float32" in bits/floatn-common.h (ICC rejects that syntax).
  CC = mpiicc
  CFLAGS = -qopenmp -O3 -Wall -std=c99 \
    -D__GNUC__=7 -D__GNUC_MINOR__=0
  LDFLAGS = -lm
else
  # GCC (default)
  CC = mpicc
  CFLAGS = -fopenmp -O3 -Wall -Wextra -std=c99
  LDFLAGS = -lm
endif

SPARSE_DIR = Sparse
HYBRIDVEC_DIR = HybridVec
SPARSE_SRC = $(SPARSE_DIR)/sparse_application.c
HYBRIDVEC_SRC = $(HYBRIDVEC_DIR)/hybrid_vec_gpu.cu

TARGETS = sparse_application hybrid_vec_gpu

# Prefer canonical name lulesh/; fall back to LULESH/ (vendor checkout name)
LULESH_DIR := $(if $(wildcard lulesh/lulesh.cc),lulesh,LULESH)
LULESH_SRC = $(LULESH_DIR)/lulesh.cc $(LULESH_DIR)/lulesh-comm.cc $(LULESH_DIR)/lulesh-viz.cc \
             $(LULESH_DIR)/lulesh-util.cc $(LULESH_DIR)/lulesh-init.cc
LULESH_CXX ?= mpicxx
LULESH_FLAGS = -O3 -g -fopenmp -I$(LULESH_DIR) -DUSE_MPI=1

# Default target
all: $(TARGETS)

# Compile sparse application (sources under Sparse/)
sparse_application: $(SPARSE_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# GPU hybrid_vec — nvcc + MPI host driver (MVAPICH/mpirun jobs).
NVCC ?= nvcc
HYBRID_VEC_NVCCFLAGS ?= -O3 -lineinfo -ccbin mpicxx --std=c++14

hybrid_vec_gpu: $(HYBRIDVEC_SRC)
	$(NVCC) $(HYBRID_VEC_NVCCFLAGS) -Xcompiler -fopenmp -lgomp $< -o $@
	cp -f $@ $(HYBRIDVEC_DIR)/$@

# LULESH 2.x app (MPI + OpenMP) for auto-tuning
$(LULESH_DIR)/lulesh2.0: $(LULESH_SRC)
	$(LULESH_CXX) $(LULESH_FLAGS) -o $@ $(LULESH_SRC) -lm

lulesh2.0: $(LULESH_DIR)/lulesh2.0

# Clean build artifacts
clean:
	rm -f $(TARGETS) $(LULESH_DIR)/lulesh2.0 *.o $(SPARSE_DIR)/*.o
	rm -f hybrid_vec_gpu $(HYBRIDVEC_DIR)/hybrid_vec_gpu

# Test compilation (does not require MPI runtime)
test-compile:
	@echo "Testing compilation environment..."
	@which $(CC) > /dev/null 2>&1 || (echo "❌ Error: $(CC) not found. Load MPI module first." && exit 1)
	@echo "✅ $(CC) found"

# Quick sanity check (run sparse_application for 1 iteration)
verify: sparse_application
	@echo "Testing sparse_application compilation (OMP_NUM_THREADS=4)..."
	@export OMP_NUM_THREADS=4 && ./sparse_application --size 256 --iterations 1


# Help target
help:
	@echo "Application Suite Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make                    - Compile sparse + hybrid_vec GPU (default)"
	@echo "  make lulesh2.0          - LULESH 2.x MPI+OpenMP (mpicxx -fopenmp)"
	@echo "  make COMPILER=intel      - Compile with Intel compiler + Intel MPI (for Nsight OpenMP)"
	@echo "  make clean               - Remove compiled files"
	@echo "  make help                - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - GCC:  module load gcc mvapich2  then  make"
	@echo "  - Intel: module load intel/compiler/64 intel/mpi/64  then  make COMPILER=intel"
	@echo "  - OpenMP and math library (-lm)"
	@echo ""
	@echo "LEAP2 HPC:"
	@echo "  GCC:   module load gcc mvapich2/gcc/64  ; make hybrid_vec_gpu"
	@echo "  Intel: module load intel/compiler/64 intel/mpi/64  ; make COMPILER=intel"
	@echo ""
	@echo "Manual Intel build (if make COMPILER=intel still fails):"
	@echo "  mpiicc -qopenmp -O3 -Wall -std=c99 -D__GNUC__=7 -D__GNUC_MINOR__=0 -o sparse_application sparse_application.c -lm"

.PHONY: all clean test-compile help lulesh2.0
