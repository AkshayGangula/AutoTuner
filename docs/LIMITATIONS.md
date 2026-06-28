# Known limitations

- **δ (GPU):** Multi-rank CUPTI refinement depends on trace coverage; rank-0-only Nsight runs may under-report other ranks until refinement runs.
- **ε (OpenMP):** Accurate NVTX-based ε requires rebuilding HybridVec with `USE_NVTX` and linking nvToolsExt; default traces use scheduling heuristics.
- **LIKWID (γ):** Requires LIKWID modules on the cluster and compatible pinning; disabled for CPU-only profiles.
- **mpiP α:** Parser guards against swapped App/MPI columns; invalid denominators fall back to safer estimates.
- **Interim vs final:** `scoring_results/` from Step 4 does not include all post-profiling refinements; use `comprehensive_results/` for papers.
- **Large artifacts:** Full `results/<job_id>/profiling/*.sqlite` and mpiP reports are not committed to git; archive externally.
