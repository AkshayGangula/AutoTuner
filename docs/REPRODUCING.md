# Reproducing an AutoTuner run

## Prerequisites

- Python 3.9+ with packages in `requirements.txt`
- SLURM cluster with CUDA/MVAPICH (or adjust `config/hpc_config.json`)
- Built benchmark binary

## 1. Configure the site

```bash
cp config/hpc_config.example.json config/hpc_config.json
```

Edit `config/hpc_config.json` so the `--system` key you pass to `master_tuner.py` matches a block with correct `partition`, `modules`, and optional `artifact_dir` / `submit_chdir`.

## 2. Build HybridVec (GPU demo)

```bash
cd applications
make hybrid_vec_gpu
# optional: install to scratch bin, e.g. cp hybrid_vec_gpu /scratch/$USER/autotuner-bin/
```

For instrumented OpenMP scheduling (ε via NVTX), rebuild with NVTX enabled in the HybridVec Makefile / `-DUSE_NVTX`.

## 3. Run the pipeline

From the repository root:

```bash
python3 scripts/master_tuner.py \
  --application hybrid_vec \
  --application-profile hybrid_vec \
  --executable /path/to/hybrid_vec_gpu \
  --system example-hpc-gpu \
  --partition gpu_a100 \
  --time-limit 00:30:00
```

Output directory: `data/experiments/hybrid_vec_<timestamp>/`.

## 4. Regenerate figures offline

```bash
python3 scripts/generate_results.py latest --system example-hpc-gpu
# or
python3 scripts/generate_results.py data/experiments/hybrid_vec_<timestamp> --system example-hpc-gpu
```

## 5. What to cite

Use `comprehensive_results/results.json` and figures under `comprehensive_results/`. Step-4 `scoring_results/` may differ before profiling refinement.

## Backfill SLURM logs

If stdout/stderr were written outside the collector search path:

```bash
python3 scripts/recollect_slurm_logs.py data/experiments/hybrid_vec_<timestamp>
```
