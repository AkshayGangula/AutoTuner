# AutoTuner

Open-source framework for MPI×OpenMP×GPU layout selection on SLURM. AutoTuner runs a two-phase study (prescreen all layouts, then mpiP / LIKWID / Nsight Systems on a subset), fuses profiler evidence into normalized axis scores (α–ε), and reports dual winners: fastest runtime and best composite balance score.

Paper: *AutoTuner: Measurement Fusion for MPI×OpenMP×GPU Layout Selection on SLURM* (The Journal of Supercomputing).

**Full HPC experiment archives** (profiler traces, SLURM logs, all campaign `results.json` files) are published on Zenodo — see [Data](#data) below. This repository contains software, site templates, benchmarks, and small example outputs.

## Repository layout

```
AutoTuner/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                    # REPRODUCING, METRICS, LIMITATIONS, LEAP2
├── config/
│   ├── profiles.yaml
│   └── hpc_config.example.json   # copy to hpc_config.json (local, not committed)
├── autotuner/               # core library
├── scripts/                 # master_tuner.py, generate_results.py
├── applications/            # HybridVec, Sparse, LULESH, miniMD sources
├── examples/                # small reviewer-facing result bundle
├── paper/                   # figure/table copies for the HybridVec case
└── data/
    └── experiments/         # gitignored; download from Zenodo
```

## Quick start

```bash
pip install -r requirements.txt
cp config/hpc_config.example.json config/hpc_config.json   # edit for your site
cd applications && make hybrid_vec_gpu
```

Example campaign on a GPU partition (`--system` must match a key in `config/hpc_config.json`):

```bash
python3 scripts/master_tuner.py \
  --application hybrid_vec \
  --application-profile hybrid_vec \
  --executable /path/to/hybrid_vec_gpu \
  --system example-hpc-gpu \
  --partition gpu_a100 \
  --account <your_allocation>
```

After jobs finish:

```bash
python3 scripts/generate_results.py data/experiments/hybrid_vec_<timestamp> --system example-hpc-gpu
```

Trust **`comprehensive_results/results.json`** for final rankings (not interim `scoring_results/`).

## Data

Full experiment trees are **not** stored in git. Download the Zenodo dataset archive and place campaigns under `data/experiments/`, or request raw traces from the corresponding author.

See `data/README.md` for layout and campaign identifiers from the manuscript.

## Documentation

- [Reproducing a run](docs/REPRODUCING.md)
- [Metric definitions](docs/METRICS.md)
- [Known limitations](docs/LIMITATIONS.md)
- [Example HPC site notes](docs/LEAP2.md)

## Example bundle

`examples/hybrid_vec_leap2_paper/` — compact `results.json`, CSV, and config snapshot for one HybridVec campaign (no profiler sqlite).

## Citation

```bibtex
@software{autotuner2026,
  title  = {AutoTuner: Measurement Fusion for MPI--OpenMP--GPU Layout Selection on SLURM},
  author = {Gangula, Akshay Reddy},
  year   = {2026},
  url    = {https://github.com/akshaygangula/AutoTuner}
}
```

Document the experiment id (e.g. `hybrid_vec_1779311216`) and cite `comprehensive_results/results.json` provenance when reproducing paper tables.

## Author

Akshay Reddy Gangula — akshaygangula1377@gmail.com — Texas State University
