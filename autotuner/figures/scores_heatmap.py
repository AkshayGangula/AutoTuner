from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from autotuner.figures.plot_helpers import epsilon_component_label


def plot_scores_heatmap(
    df, output_dir: Path, logger: Any, cpu_only: bool = False
) -> None:
    """
    Save the heatmap for heuristic components:
    α_comm, β_thread, γ_locality, [δ_gpu if not cpu_only], ε_openmp, heuristic_score.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    heatmap_cols = [
        "alpha_comm",
        "beta_thread",
        "gamma_locality",
    ]
    if not cpu_only:
        heatmap_cols.append("delta_gpu")
    heatmap_cols.extend(["epsilon_openmp", "heuristic_score"])
    available_cols = [col for col in heatmap_cols if col in df.columns]
    if len(available_cols) != len(heatmap_cols):
        missing = set(heatmap_cols) - set(available_cols)
        logger.warning(f"⚠️  Missing columns for heatmap: {missing}")

    heatmap_data = df.set_index("config_name")[available_cols]
    column_labels = {
        "alpha_comm": "α Comm",
        "beta_thread": "β Thread",
        "gamma_locality": "γ Locality",
        "delta_gpu": "δ GPU",
        "epsilon_openmp": epsilon_component_label(df),
        "heuristic_score": "HEURISTIC",
    }
    heatmap_data.columns = [column_labels.get(col, col) for col in available_cols]

    heatmap_data = heatmap_data.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)

    # Absolute 0–1 scale for color (matches cell numbers). Multi-rank δ GPU is
    # legitimately low (~0.01) and will look red; that is measurement, not a bug.
    sns.heatmap(
        heatmap_data,
        annot=heatmap_data,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=1,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score (0–1, higher is better)"},
        annot_kws={"size": 11},
    )
    norm_note = "\n(color = absolute score 0–1; δ GPU low on multi-rank is expected with rank-0 CUPTI)"

    nmet = len(available_cols) - 1  # minus heuristic_score
    title = f"{nmet} Heuristic Components + Composite Score Heatmap"
    if cpu_only:
        title += "\n(CPU-only: δ GPU omitted)"
    if "gamma_locality" in df.columns and (df["gamma_locality"] == 0).all():
        title += "\n(γ Locality = 0: LIKWID not used or locality disabled for this profile)"
    title += norm_note
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    heatmap_path = output_dir / "scores_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"✅ Saved heatmap: {heatmap_path}")
    plt.close()

