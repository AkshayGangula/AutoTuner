# HybridVec paper example bundle

Small artifact bundle: final scores and exports (no SLURM logs or Nsight sqlite).

| Path | Description |
|------|-------------|
| `comprehensive_results/results.json` | Final ranked configurations and provenance |
| `comprehensive_results/configuration_scores.csv` | Tabular export |
| `config_snapshot.json` | CLI / system snapshot (paths use `<user>` placeholders) |

Full experiment trees for all manuscript campaigns are on **Zenodo**; see `data/README.md` at the repository root.

Regenerate figures when a full experiment directory is available:

```bash
python3 scripts/generate_results.py data/experiments/hybrid_vec_<timestamp> --system example-hpc-gpu
```
