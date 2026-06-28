"""Per-app CLI defaults and SLURM / mpiP / LIKWID quirks."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

_DEFAULT_HYBRID_VEC_CLI = "--size 1024 --iterations 100"


def apply_application_cli_defaults(args: argparse.Namespace) -> None:
    """Mutate ``args`` for app-specific CLI conventions (e.g. LULESH uses -s/-i, not --size)."""
    if args.application == "lulesh":
        if args.arguments.strip() == _DEFAULT_HYBRID_VEC_CLI:
            args.arguments = "-s 22 -i 40"
            print("📋 LULESH: using default args '-s 22 -i 40' (override with --arguments)")
        if args.phase1_arguments is None:
            args.phase1_arguments = "-s 18 -i 15"


def effective_mpip_for_job(
    application_name: str, nodes_for_job: int, global_enable_mpip: bool
) -> Tuple[bool, Optional[str]]:
    """
    LULESH + mpiP PMPI on multi-node issue avoidance.
    Keep mpiP for single-node LULESH; use Nsight for α on multi-node LULESH.
    """
    if not global_enable_mpip:
        return False, None
    if application_name == "lulesh" and nodes_for_job > 1:
        return False, (
            "  mpiP off for this job (LULESH, >1 node): avoids PMPI multi-node crash; "
            "α from Nsight if traced."
        )
    return True, None


def likwid_rank0_only_for_job(application_name: str, nodes_for_job: int) -> Tuple[bool, Optional[str]]:
    if application_name == "lulesh" and nodes_for_job > 1:
        return True, (
            "  LIKWID rank-0 only (LULESH, >1 node): Run 1b/1c use likwid on MPI rank 0 only; "
            "γ from rank-0 counters."
        )
    return False, None


def phase1_likwid_time_warning(
    application_name: str, enable_likwid: bool, phase1_time_limit: Optional[str]
) -> Optional[str]:
    """Return a warning string if Phase-1 wall time is likely too short for LIKWID-heavy apps."""
    if not phase1_time_limit or not enable_likwid:
        return None
    if application_name not in ("lulesh", "hybrid_vec"):
        return None
    try:
        parts = phase1_time_limit.strip().split(":")
        mins = int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return None
    if application_name == "lulesh" and mins < 15:
        return (
            "  ⚠️  LULESH Phase-1 runs Run 1a plus LIKWID (1b NUMA + 1c MEM on every rank); "
            "under ~15 minutes often hits TIME LIMIT. Prefer --phase1-time-limit 00:15:00 or 00:20:00."
        )
    if application_name == "hybrid_vec" and mins < 20:
        return (
            "  ⚠️  hybrid_vec Phase-1 runs Run 1a + LIKWID (1b/1c); GPU MPI layouts often need "
            "≥20 minutes wall time. Prefer --phase1-time-limit 00:20:00 or 00:25:00."
        )
    return None
