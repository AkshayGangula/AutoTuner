"""
Stacked bar chart: wall time split into useful compute, MPI communication, thread stall/sync.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_runtime_stacked_breakdown(df, output_dir: Path, logger: Any) -> None:
    need = {"config_name", "runtime_sec", "mpi_comm_time", "thread_stall_time"}
    if not need.issubset(df.columns):
        logger.warning("⚠️  runtime_stacked_breakdown: missing columns, skipping")
        return
    if len(df) < 1:
        return

    work = df.copy()
    rt = pd.to_numeric(work["runtime_sec"], errors="coerce").fillna(0.0).clip(lower=0.0)
    mpi = pd.to_numeric(work["mpi_comm_time"], errors="coerce").fillna(0.0).clip(lower=0.0)
    stall = pd.to_numeric(work["thread_stall_time"], errors="coerce").fillna(0.0).clip(lower=0.0)

    mpi_c = np.zeros(len(work), dtype=float)
    if "mpip_max_mpi_wall_fraction" in work.columns:
        wall_frac = pd.to_numeric(work["mpip_max_mpi_wall_fraction"], errors="coerce").fillna(0.0)
        use_mpip = wall_frac > 0.001
        mpi_c = np.where(use_mpip, np.minimum(wall_frac.values * rt.values, rt.values), mpi_c)
    mpi_c = np.where(mpi_c <= 1e-9, np.minimum(mpi.values, rt.values), mpi_c)
    stall_c = np.minimum(stall.values, np.maximum(0.0, rt.values - mpi_c))
    useful = np.maximum(0.0, rt.values - mpi_c - stall_c)

    # If no profiling, entire bar is "useful" (unknown breakdown)
    no_prof = (mpi_c <= 1e-9) & (stall_c <= 1e-9)
    useful = np.where(no_prof, rt.values, useful)
    mpi_c = np.where(no_prof, 0.0, mpi_c)
    stall_c = np.where(no_prof, 0.0, stall_c)

    x = np.arange(len(work))
    names = work["config_name"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(work) * 1.2), 6))
    b1 = ax.bar(x, useful, label="Useful / other (incl. unprofiled)", color="#2ecc71", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x, mpi_c, bottom=useful, label="MPI communication", color="#e74c3c", edgecolor="black", linewidth=0.4)
    b3 = ax.bar(x, stall_c, bottom=useful + mpi_c, label="Thread stall / sync", color="#f39c12", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    pos_rt = rt.values[rt.values > 0]
    use_log = len(pos_rt) > 1 and (pos_rt.max() / max(pos_rt.min(), 1e-9)) >= 50.0
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel("Time (seconds, log scale)")
    else:
        ax.set_ylabel("Time (seconds)")
    title = "Runtime breakdown by configuration"
    if "mpip_max_mpi_wall_fraction" in work.columns and (
        pd.to_numeric(work["mpip_max_mpi_wall_fraction"], errors="coerce").fillna(0.0) > 0.001
    ).any():
        title += " (MPI slice from mpiP wall fraction when available)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "runtime_breakdown_stacked.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved stacked runtime breakdown: {out}")
