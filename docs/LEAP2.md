# Example HPC site notes

## Paths

| Item | Typical path |
|------|----------------|
| Repo clone | `~/AutoTuner` |
| Experiment tree | `~/AutoTuner/data/experiments/<app>_<timestamp>/` |
| Installed binary (optional) | `/scratch/$USER/autotuner-bin/hybrid_vec_gpu` |

The **binary** may live on scratch for convenience; **experiment artifacts** (profiling, LIKWID, mpiP, `slurm-*.out`) are written under the experiment directory when `artifact_dir` is unset (default for `example-hpc-gpu` in `config/hpc_config.json`).

## Where artifacts go (repo vs scratch)

| `hpc_config.json` keys | Profiling / LIKWID / mpiP | SLURM logs |
|------------------------|---------------------------|------------|
| **None** (`example-hpc-gpu` default) | `data/experiments/<exp>/profiling/<job_id>/` | `data/experiments/<exp>/slurm-<job_id>.out` |
| `artifact_dir` + `submit_chdir` + `slurm_log_dir` | `/scratch/$USER/autotuner-data/<exp>/profiling/<job_id>/` | `/scratch/$USER/slurm-<job_id>.out` |

Use scratch keys only if large Nsight traces on shared filesystems cause slowdowns or stale-file errors on some GPU nodes.

## Config snippet

`example-hpc-gpu` needs GPU partition modules (gcc, mvapich2, cuda) and `mpip_mvapich_srun: true` for mpiP Run 1a. Use `example-hpc-cpu` for CPU-only apps (`partition: cpu_shared`, no CUDA module).

## Modules

Load the same modules in job scripts as on the login node used for `make hybrid_vec_gpu`. Mismatched CUDA/MVAPICH versions produce empty GPU traces.

## Node exclude

If specific nodes misbehave, set `sbatch_exclude` in `hpc_config.json` (e.g. a hostname or node list). Remove the exclude after admins repair the node.

## Move old scratch artifacts into the experiment folder

```bash
EXP=hybrid_vec_<timestamp>
mkdir -p ~/AutoTuner/data/experiments/$EXP/profiling
rsync -av /scratch/$USER/autotuner-data/$EXP/profiling/ \
  ~/AutoTuner/data/experiments/$EXP/profiling/
```

## Rebuild on cluster

```bash
cd ~/AutoTuner/applications && make hybrid_vec_gpu
cp HybridVec/hybrid_vec_gpu /scratch/$USER/autotuner-bin/
```

## Verify next run uses the experiment dir

```bash
grep EXPERIMENT_DIR ~/AutoTuner/data/experiments/hybrid_vec_*/scripts/*.slurm | head -3
# Should show .../data/experiments/hybrid_vec_<ts>, not /scratch/.../autotuner-submit/...
```
