"""Default executables (under ``applications/``) and MPI×OpenMP configuration sweeps per app."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from autotuner.core.configuration_generator import ConfigurationGenerator

logger = logging.getLogger(__name__)

# Relative to repo ``applications/`` after ``make`` (see applications/Makefile).
APPLICATION_EXECUTABLE_RELATIVE: Dict[str, str] = {
    "sparse_matrix": "sparse_application",
    "hybrid_vec": "hybrid_vec_gpu",
    "lulesh": "lulesh/lulesh2.0",
}

# Drop layouts with too many MPI ranks per node on typical single-GPU nodes.
_HYBRID_VEC_MAX_RANKS_PER_NODE = 16


def filter_gpu_mpi_configurations(
    configurations: List[Any],
    max_ranks_per_node: int = _HYBRID_VEC_MAX_RANKS_PER_NODE,
) -> List[Any]:
    filtered: List[Any] = []
    for c in configurations:
        m = int(getattr(c, "mpi_ranks_per_node", 0) or 0)
        if m > max_ranks_per_node:
            continue
        filtered.append(c)
    return filtered if filtered else configurations


def _filter_hybrid_vec_configs(
    configurations: List[Any], *, label: str, num_nodes: Optional[int] = None
) -> List[Any]:
    n_before = len(configurations)
    out = filter_gpu_mpi_configurations(configurations)
    if n_before != len(out):
        extra = f" (num_nodes={num_nodes})" if num_nodes is not None else ""
        logger.info(
            "hybrid_vec%s: %d -> %d configs after filter "
            "(drop mpi_ranks_per_node > %d for GPU stability).",
            f" ({label}){extra}" if label else "",
            n_before,
            len(out),
            _HYBRID_VEC_MAX_RANKS_PER_NODE,
        )
    return out


def resolve_default_executable_path(applications_dir: Path, application: str) -> Optional[Path]:
    """Return expected binary path, or None if this app has no registered default."""
    rel = APPLICATION_EXECUTABLE_RELATIVE.get(application)
    if not rel:
        return None
    return applications_dir / rel


def build_configurations_for_application(
    *,
    application_name: str,
    config_generator: ConfigurationGenerator,
    num_nodes: int,
    pilot_only: bool,
) -> List[Any]:
    if pilot_only:
        if application_name == "lulesh":
            all_l = config_generator.generate_lulesh_configurations(num_nodes)
            return all_l[: min(4, len(all_l))] if len(all_l) > 4 else all_l
        if application_name == "minimd":
            all_m = config_generator.generate_minimd_configurations()
            return all_m[: min(4, len(all_m))] if len(all_m) > 4 else all_m
        rec = config_generator.get_recommended_configurations()
        if application_name == "hybrid_vec":
            return _filter_hybrid_vec_configs(rec, label="pilot", num_nodes=num_nodes)
        return rec

    if application_name == "lulesh":
        return config_generator.generate_lulesh_configurations(num_nodes)
    if application_name == "minimd":
        return config_generator.generate_minimd_configurations()

    paper_configs = config_generator.generate_paper_configurations()
    numa_configs = config_generator.generate_numa_aware_configurations()
    all_configs = paper_configs + numa_configs
    configurations: List[Any] = []
    seen_names = set()
    for config in all_configs:
        if config.name not in seen_names:
            configurations.append(config)
            seen_names.add(config.name)

    if application_name == "hybrid_vec":
        return _filter_hybrid_vec_configs(configurations, label="full")

    return configurations
