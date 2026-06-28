# Experiment data

Full auto-tuning runs are **not** committed to this repository.

## Zenodo archive

Download the HPC campaign dataset from Zenodo (see the Data availability statement in the manuscript or the repository root README). Unpack so each campaign appears as:

```
data/experiments/<application>_<timestamp>/
├── generated_configurations.json
├── phase1_results.json
├── phase2_job_ids.json
├── scripts/
├── results/<job_id>/
└── comprehensive_results/
    └── results.json          # authoritative rankings for papers
```

## Campaign identifiers (manuscript Section 5)

| Workload | $R$ | Directories |
|----------|-----|-------------|
| HybridVec | 5 | `hybrid_vec_1779311216`, `hybrid_vec_1779461252`, `hybrid_vec_1779461256`, `hybrid_vec_1779462282`, `hybrid_vec_1779462285` |
| sparse_matrix | 3 | `sparse_matrix_1779312778`, `sparse_matrix_1779460227`, `sparse_matrix_1779460230` |
| LULESH | 3 | `lulesh_1779313542`, `lulesh_1779458085`, `lulesh_1779458193` |
| miniMD | 3 | `minimd_1779312844`, `minimd_1779459466`, `minimd_1779459468` |

Re-fuse after download:

```bash
python3 scripts/generate_results.py data/experiments/<campaign_id> --system example-hpc-gpu
```

Use `--system example-hpc-cpu` for CPU workloads on the `cpu_shared` partition.

Legacy path `auto_tuning_experiments/` is still discovered by `scripts/generate_results.py` if present.
