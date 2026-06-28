"""LULESH-specific tuning and stdout / job-script parsing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

# LULESH — optional "(s)" / "(z/s)" in stdout (values may be scientific, e.g. 1e+02)
_RE_LULESH_ELAPSED = re.compile(
    r"Elapsed\s+time\s*=\s*([\d.eE+-]+)(?:\s*\(\s*s\s*\))?", re.IGNORECASE
)
_RE_LULESH_FOM = re.compile(
    r"FOM\s*=\s*([\d.eE+-]+)(?:\s*\(\s*z\s*/\s*s\s*\))?", re.IGNORECASE
)
_RE_LULESH_CMD_S_I = re.compile(
    r"lulesh2\.0\b[^;\n]*?-s\s+(\d+)\s+-i\s+(\d+)", re.IGNORECASE
)


def parse_lulesh_runtime_throughput(text: str) -> Tuple[float, float]:
    """Last occurrence wins (matches final 'Run completed' block)."""
    runtime = 0.0
    throughput = 0.0
    for m in _RE_LULESH_ELAPSED.finditer(text):
        try:
            runtime = float(m.group(1))
        except ValueError:
            pass
    for m in _RE_LULESH_FOM.finditer(text):
        try:
            throughput = float(m.group(1))
        except ValueError:
            pass
    return runtime, throughput


def parse_lulesh_mesh_iter_from_job_script(path: Path) -> Tuple[float, float]:
    """Read `-s` / `-i` from an embedded `lulesh2.0 ...` line in a SLURM job script."""
    if not path.exists():
        return (0.0, 0.0)
    text = path.read_text(errors="replace")
    m = _RE_LULESH_CMD_S_I.search(text)
    if not m:
        return (0.0, 0.0)
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return (0.0, 0.0)


def lulesh_throughput_z_per_s_from_mesh(
    mesh_s: float, iterations: float, runtime_sec: float
) -> float:
    """When stdout has no FOM (e.g. `-q`), approximate z/s as mesh³×iter / wall time."""
    if mesh_s <= 0 or iterations <= 0 or runtime_sec <= 0:
        return 0.0
    return (mesh_s**3) * iterations / runtime_sec


def reference_mpi_rank_count_from_configs(
    configurations: List[Any], default_num_nodes: int
) -> int:
    """Max total MPI ranks across configs (for LULESH -s scaling reference)."""
    ref = 0
    for c in configurations:
        n = getattr(c, "num_nodes", None) or default_num_nodes
        ref = max(ref, int(n) * int(c.mpi_ranks_per_node))
    return ref


def scale_arguments_for_equal_total_zones(
    arguments: str, total_ranks: int, reference_ranks: int
) -> str:
    """Scale LULESH's -s N so all configs solve the same total problem size.

    LULESH's -s N is mesh size *per MPI rank* (N³ zones per rank).
    Total zones = N³ × total_ranks.  To keep this constant across configs:

        scaled_s = round( (N³ × reference_ranks / total_ranks) ^ (1/3) )

    If -s is not present in arguments, arguments are returned unchanged.
    """
    m = re.search(r"-s\s+(\d+)", arguments)
    if not m or reference_ranks <= 0 or total_ranks <= 0:
        return arguments
    base_s = int(m.group(1))
    total_zones = (base_s**3) * reference_ranks
    scaled_s = max(1, round((total_zones / total_ranks) ** (1.0 / 3.0)))
    scaled = re.sub(r"-s\s+\d+", f"-s {scaled_s}", arguments)
    if scaled_s != base_s:
        logger.info(
            "  LULESH problem-size scaling: %s ranks → -s %s → -s %s "
            "(%s zones, matching %s-rank reference of %s zones)",
            total_ranks,
            base_s,
            scaled_s,
            f"{scaled_s**3 * total_ranks:,}",
            reference_ranks,
            f"{base_s**3 * reference_ranks:,}",
        )
    return scaled
