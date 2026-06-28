from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autotuner.figures.plot_helpers import mpi_comm_seconds_for_plot, openmp_panel_title


def plot_profiling_metrics(df, output_dir: Path, logger: Any) -> None:
    """
    Save profiling metric plot (1x3):
    - MPI communication time (mpiP wall fraction × runtime when available)
    - CPU utilization (cpu_utilization as %)
    - CPU scheduling (trace-based) or OpenMP efficiency
    """
    work = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Profiling Metrics by Configuration",
        fontsize=14,
        fontweight="bold",
    )

    mpi_comm_data = mpi_comm_seconds_for_plot(work)
    uses_mpip = (
        "mpip_max_mpi_wall_fraction" in work.columns
        and (pd.to_numeric(work["mpip_max_mpi_wall_fraction"], errors="coerce").fillna(0) > 0.001).any()
    )

    bars1 = axes[0].bar(
        work["config_name"],
        mpi_comm_data,
        color="#3498db",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Time (seconds)")
    mpi_title = "MPI Communication Time"
    if uses_mpip:
        mpi_title += " (mpiP wall fraction × runtime)"
    axes[0].set_title(mpi_title, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)
    if mpi_comm_data.max() > 0:
        for bar, val in zip(bars1, mpi_comm_data):
            if val > 0:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mpi_comm_data.max() * 0.01,
                    f"{val:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    else:
        axes[0].text(
            0.5, 0.5, "No data available",
            transform=axes[0].transAxes, ha="center", va="center",
            fontsize=12, style="italic", color="gray",
        )

    if "cpu_utilization" not in work.columns:
        work["cpu_utilization"] = 0.0
    cpu_util_data = work["cpu_utilization"].fillna(0.0) * 100
    cpu_util_data = cpu_util_data.replace([np.inf, -np.inf], 0.0).clip(0, 100)
    bars2 = axes[1].bar(
        work["config_name"],
        cpu_util_data,
        color="#9b59b6",
        edgecolor="black",
        alpha=0.7,
    )
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].set_title("CPU Utilization (Nsight)", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)
    if cpu_util_data.max() > 0:
        for bar, val in zip(bars2, cpu_util_data):
            if val > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

    openmp_col = (
        "openmp_work_efficiency_display"
        if "openmp_work_efficiency_display" in work.columns
        else "openmp_work_efficiency"
    )
    if openmp_col not in work.columns:
        work[openmp_col] = 0.0
    openmp_eff_data = work[openmp_col].fillna(0.0) * 100
    openmp_eff_data = openmp_eff_data.replace([np.inf, -np.inf], 0.0).clip(0, 100)

    bars3 = axes[2].bar(
        work["config_name"],
        openmp_eff_data,
        color="#27ae60",
        edgecolor="black",
        alpha=0.7,
    )
    axes[2].set_xlabel("Configuration")
    axes[2].set_ylabel("Score (%)")
    axes[2].set_title(openmp_panel_title(work), fontweight="bold")
    axes[2].set_ylim(0, 100)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(axis="y", alpha=0.3)
    if openmp_eff_data.max() > 0:
        for bar, val in zip(bars3, openmp_eff_data):
            if val > 0:
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

    plt.tight_layout()
    profiling_path = output_dir / "profiling_metrics.png"
    plt.savefig(profiling_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"✅ Saved profiling metrics: {profiling_path}")
    plt.close()
