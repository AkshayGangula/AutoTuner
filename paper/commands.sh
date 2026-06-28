#!/usr/bin/env bash
# Paper reproduction commands (run from AutoTuner repo root on your HPC)
set -euo pipefail

cp config/hpc_config.example.json config/hpc_config.json
# Edit config/hpc_config.json for your allocation, partition, and scratch paths.

cd applications
make hybrid_vec_gpu
cd ..

python3 scripts/master_tuner.py \
  --application hybrid_vec \
  --application-profile hybrid_vec \
  --executable "/scratch/${USER}/autotuner-bin/hybrid_vec_gpu" \
  --system example-hpc-gpu \
  --partition gpu_a100

# After SLURM jobs complete:
python3 scripts/generate_results.py latest --system example-hpc-gpu

# Copy submission assets (adjust EXP if not using 'latest'):
EXP="$(ls -td data/experiments/hybrid_vec_* 2>/dev/null | head -1)"
if [[ -n "${EXP}" && -d "${EXP}/comprehensive_results" ]]; then
  cp "${EXP}/comprehensive_results/"*.png paper/figures/
  cp "${EXP}/comprehensive_results/"*.{csv,json} paper/tables/
fi
