# Benchmark applications

| Directory | Role |
|-----------|------|
| `HybridVec/` | First-party GPU demo (`hybrid_vec_gpu.cu`) |
| `Sparse/` | First-party CPU sparse kernel (`sparse_application.c`) |
| `lulesh/` | LLNL LULESH 2.x vendor tree (stored as `LULESH/` on disk) |
| `minimd/` | Mantevo miniMD vendor tree (stored as `miniMD/` on disk) |

## Build

```bash
make                  # sparse_application, hybrid_vec_gpu
make lulesh2.0        # uses lulesh/ sources (LULESH/ on case-insensitive macOS)
```

On **Linux (LEAP2)**, optional canonical names:

```bash
ln -sfn LULESH lulesh
ln -sfn miniMD minimd
```

Vendor trees are full upstream checkouts (not submodules in this repo). For a slimmer clone, replace with git submodules pointing to LLNL LULESH and Mantevo miniMD.
