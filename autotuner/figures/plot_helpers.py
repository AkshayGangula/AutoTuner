"""Shared helpers for auto-tuning result figures."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mpi_comm_seconds_for_plot(df: pd.DataFrame) -> pd.Series:
    """
    MPI time (seconds) for bar charts: prefer mpiP wall fraction × runtime when available.
    """
    rt = pd.to_numeric(df.get("runtime_sec"), errors="coerce").fillna(0.0).clip(lower=0.0)
    mpi = pd.to_numeric(df.get("mpi_comm_time"), errors="coerce").fillna(0.0).clip(lower=0.0)

    if "mpip_max_mpi_wall_fraction" in df.columns:
        wall_frac = pd.to_numeric(df["mpip_max_mpi_wall_fraction"], errors="coerce").fillna(0.0)
        use_mpip = wall_frac > 0.001
        from_mpip = np.minimum(wall_frac.values * rt.values, rt.values)
        mpi_vals = np.where(use_mpip, from_mpip, np.minimum(mpi.values, rt.values))
        return pd.Series(mpi_vals, index=df.index)

    return np.minimum(mpi, rt)


def epsilon_component_label(df: pd.DataFrame) -> str:
    """Heatmap/dashboard label for ε when OpenMP is not instrumented in the binary."""
    if "openmp_instrumented" in df.columns:
        if not df["openmp_instrumented"].fillna(False).astype(bool).any():
            return "ε CPU sched (trace)"
    return "ε OpenMP"


def openmp_panel_title(df: pd.DataFrame) -> str:
    if "openmp_instrumented" in df.columns:
        if df["openmp_instrumented"].fillna(False).astype(bool).any():
            return "OpenMP Work Efficiency"
    if "openmp_work_efficiency" in df.columns and (df["openmp_work_efficiency"].fillna(0) > 0.01).any():
        return "CPU Scheduling (trace, not OMPT)"
    return "OpenMP Efficiency (est. from runtime when not traced)"


def highlight_mask(df: pd.DataFrame, rank_col: str) -> pd.Series:
    """True where rank_col == 1 (best for that ranking)."""
    if rank_col not in df.columns:
        return pd.Series(False, index=df.index)
    ranks = pd.to_numeric(df[rank_col], errors="coerce")
    return ranks == ranks.min()
