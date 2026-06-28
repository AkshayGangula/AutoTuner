# Heuristic metrics (α–ε)

AutoTuner ranks MPI×OpenMP×GPU configurations using five normalized components. Final scores live in `comprehensive_results/results.json` after Step 7 / `generate_results.py`.

| Symbol | Name | Typical source |
|--------|------|----------------|
| **α** | Communication efficiency | mpiP (App vs MPI time); refined when columns are sane |
| **β** | Thread / rank efficiency | mpiP, rank balance |
| **γ** | Memory locality | LIKWID (when `--enable-likwid` / profile allows) |
| **δ** | GPU utilization | Nsight Systems + optional multi-rank CUPTI refinement (`autotuner/core/profiling_refinement.py`) |
| **ε** | OpenMP scheduling | Nsight trace (NVTX ranges if app built with `USE_NVTX`; else trace heuristics) |

## Runtime vs heuristic winner

- **Fastest wall clock** — `runtime_sec` / throughput from SLURM stdout (HybridVec: parsed GFLOPS line).
- **Best heuristic** — weighted composite of α–ε; can disagree with raw speed (e.g. better GPU balance at slightly lower GFLOPS).

## Implementation pointers

- Scoring: `autotuner/core/heuristic_scoring.py`
- Extraction: `autotuner/core/metrics_extractor.py`
- GPU refinement: `autotuner/core/profiling_refinement.py`
- Profiles (CPU-only, LIKWID): `config/profiles.yaml`
