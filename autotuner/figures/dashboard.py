from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autotuner.figures.plot_helpers import epsilon_component_label, highlight_mask


def plot_auto_tuning_dashboard(
    df, output_dir: Path, logger: Any, cpu_only: bool = False
) -> None:
    """
    Save the main 2x2 dashboard plot:
    - Runtime by configuration
    - Heuristic score by configuration
    - Speedup comparison (vs fastest runtime; log scale when span is large)
    - Heuristic components: α, β, γ, ε; omit δ GPU when cpu_only (avoids empty column).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MPI+OpenMP Auto-Tuning Results Dashboard",
        fontsize=14,
        fontweight="bold",
    )

    work = df.reset_index(drop=True)
    fastest_mask = highlight_mask(work, "rank")
    best_heuristic_mask = highlight_mask(work, "heuristic_rank")

    # Runtime — highlight fastest (wall clock)
    ax1 = axes[0, 0]
    colors_rt = [
        "#27ae60" if fastest_mask.iloc[i] else "#3498db"
        for i in range(len(work))
    ]
    bars = ax1.bar(work["config_name"], work["runtime_sec"], color=colors_rt, edgecolor="black")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title("Runtime by Configuration (green = fastest)", fontweight="bold")
    max_runtime = float(work["runtime_sec"].max()) if len(work) else 0.0
    for bar, val in zip(bars, work["runtime_sec"]):
        if val > 0 and max_runtime > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_runtime * 0.01,
                f"{val:.3f}s" if val < 1.0 else f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Heuristic score — highlight best heuristic rank (may differ from fastest)
    ax2 = axes[0, 1]
    colors_h = [
        "#f39c12" if best_heuristic_mask.iloc[i] else "#95a5a6"
        for i in range(len(work))
    ]
    if "heuristic_score_10" in work.columns:
        hvals = work["heuristic_score_10"]
        ax2.set_ylabel("Heuristic (0–10)")
        ax2.set_ylim(0, 11)
    else:
        hvals = work["heuristic_score"]
        ax2.set_ylabel("Heuristic Score (0–1)")
        ax2.set_ylim(0, 1.1)
    bars = ax2.bar(work["config_name"], hvals, color=colors_h, edgecolor="black")
    ax2.set_xlabel("Configuration")
    ax2.set_title("Heuristic Score (orange = best heuristic)", fontweight="bold")

    # Slowdown vs fastest (1.0 on winner) — easier to read than tiny "performance" fractions
    ax3 = axes[1, 0]
    if "slowdown_vs_fastest" in work.columns:
        speed_vals = pd.to_numeric(work["slowdown_vs_fastest"], errors="coerce").fillna(1.0)
        speed_vals = speed_vals.clip(lower=1.0)
        ylabel = "Slowdown vs fastest (1.0 = fastest)"
        title = "Wall-Clock Slowdown vs Fastest (1×48)"
        use_log = True
    elif "speedup_vs_fastest" in work.columns:
        speed_vals = pd.to_numeric(work["speedup_vs_fastest"], errors="coerce").fillna(0.0)
        ylabel = "Performance vs fastest (1.0 = best)"
        title = "Performance vs Fastest Configuration"
        use_log = len(speed_vals[speed_vals > 0]) > 1 and (
            speed_vals.max() / max(speed_vals[speed_vals > 0].min(), 1e-9) >= 20.0
        )
    else:
        speed_vals = pd.to_numeric(work["speedup"], errors="coerce").fillna(0.0)
        ylabel = "Speedup (vs median runtime)"
        title = "Speedup Comparison"
        use_log = False

    colors_sp = [
        "#27ae60" if fastest_mask.iloc[i] else "#95a5a6"
        for i in range(len(work))
    ]
    bars = ax3.bar(work["config_name"], speed_vals, color=colors_sp, edgecolor="black")
    if "slowdown_vs_fastest" in work.columns:
        ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    else:
        ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Configuration")
    ax3.set_ylabel(ylabel)
    ax3.set_title(title, fontweight="bold")
    if use_log:
        ax3.set_yscale("log")

    for bar, val in zip(bars, speed_vals):
        if val <= 0:
            continue
        if "slowdown_vs_fastest" in work.columns:
            label = f"{val:.0f}×" if val >= 10 else f"{val:.1f}×"
        elif "speedup_vs_fastest" in work.columns:
            label = f"{val:.3f}×"
        else:
            label = f"{val:.1f}×"
        yoff = bar.get_height() * (1.08 if ax3.get_yscale() == "log" else 1.0)
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            yoff,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Heuristic components
    ax4 = axes[1, 1]
    x = np.arange(len(work))
    eps_label = epsilon_component_label(work)
    comp_specs = [
        ("alpha_comm", "α Comm", "#3498db"),
        ("beta_thread", "β Thread", "#9b59b6"),
        ("gamma_locality", "γ Locality", "#27ae60"),
    ]
    if not cpu_only:
        comp_specs.append(("delta_gpu", "δ GPU", "#f39c12"))
    comp_specs.append(("epsilon_openmp", eps_label, "#e74c3c"))
    comp_specs = [(c, l, cr) for c, l, cr in comp_specs if c in work.columns]
    ncomp = len(comp_specs)
    width = 0.8 / max(ncomp, 1)
    for i, (col, lab, colr) in enumerate(comp_specs):
        off = (i - (ncomp - 1) / 2.0) * width
        ax4.bar(x + off, work[col], width, label=lab, color=colr, edgecolor="black")
    comp_title = (
        f"{ncomp} Heuristic Score Components (CPU-only, δ GPU omitted)"
        if cpu_only
        else f"{ncomp} Heuristic Score Components"
    )
    ax4.set_xlabel("Configuration", fontweight="bold")
    ax4.set_ylabel("Component Score (0-1)", fontweight="bold")
    ax4.set_title(comp_title, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(work["config_name"])
    ax4.legend(loc="upper right", fontsize=7)
    ax4.set_ylim(0, 1.2)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    dashboard_path = output_dir / "auto_tuning_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"✅ Saved dashboard: {dashboard_path}")
    plt.close()
