"""
Runtime-vs-heuristic rank agreement plot.
Shows where heuristic ranking disagrees with measured runtime ranking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rank_agreement(df, output_dir: Path, logger: Any) -> None:
    need = {"config_name"}
    if not need.issubset(df.columns):
        logger.warning("⚠️  rank_agreement_plot: missing config_name, skipping")
        return
    if len(df) < 2:
        logger.warning("⚠️  rank_agreement_plot: need at least 2 configs, skipping")
        return

    work = df.copy()
    if "rank" in work.columns:
        runtime_rank = pd.to_numeric(work["rank"], errors="coerce")
    elif "runtime_sec" in work.columns:
        runtime_rank = pd.to_numeric(work["runtime_sec"], errors="coerce").rank(
            ascending=True, method="min"
        )
    else:
        logger.warning("⚠️  rank_agreement_plot: missing rank/runtime_sec, skipping")
        return

    if "heuristic_rank" in work.columns:
        heuristic_rank = pd.to_numeric(work["heuristic_rank"], errors="coerce")
    elif "heuristic_score" in work.columns:
        heuristic_rank = pd.to_numeric(work["heuristic_score"], errors="coerce").rank(
            ascending=False, method="min"
        )
    else:
        logger.warning("⚠️  rank_agreement_plot: missing heuristic_rank/heuristic_score, skipping")
        return

    runtime_rank = runtime_rank.fillna(0).astype(int)
    heuristic_rank = heuristic_rank.fillna(0).astype(int)
    if (runtime_rank <= 0).any() or (heuristic_rank <= 0).any():
        logger.warning("⚠️  rank_agreement_plot: invalid rank values, skipping")
        return

    delta = (heuristic_rank - runtime_rank).astype(int)
    max_rank = int(max(runtime_rank.max(), heuristic_rank.max()))
    lim = max_rank + 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        runtime_rank,
        heuristic_rank,
        c=np.abs(delta),
        cmap="YlOrRd",
        s=180,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    ax.plot([0.5, lim], [0.5, lim], "k--", linewidth=1.5, alpha=0.8, label="Perfect agreement")

    for i, row in work.reset_index(drop=True).iterrows():
        ax.annotate(
            f"{row['config_name']} (Δ={int(delta.iloc[i]):+d})",
            (float(runtime_rank.iloc[i]), float(heuristic_rank.iloc[i])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("|rank disagreement|", rotation=90)
    ax.set_xlim(0.5, lim)
    ax.set_ylim(0.5, lim)
    ax.set_xticks(range(1, max_rank + 1))
    ax.set_yticks(range(1, max_rank + 1))
    ax.set_xlabel("Runtime rank (1 = fastest)")
    ax.set_ylabel("Heuristic rank (1 = highest score)")
    ax.set_title("Rank agreement: measured runtime vs heuristic score", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = output_dir / "rank_agreement_runtime_vs_heuristic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved rank-agreement plot: {out}")
