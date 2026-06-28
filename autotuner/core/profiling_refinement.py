"""Refine δ (GPU) and ε (OpenMP) from Nsight, mpiP, and wall time."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def mpi_comm_seconds_for_gpu(
    application_runtime: float,
    mpi_comm_time: float = 0.0,
    mpip_wall_fraction: float = 0.0,
) -> float:
    """MPI seconds for GPU hybrid denominator — prefer mpiP wall fraction × app wall."""
    app_rt = max(float(application_runtime or 0.0), 0.0)
    frac = max(0.0, min(1.0, float(mpip_wall_fraction or 0.0)))
    if app_rt > 0 and frac > 0.001:
        return min(app_rt, frac * app_rt)
    mct = max(0.0, float(mpi_comm_time or 0.0))
    if app_rt > 0:
        return min(mct, app_rt)
    return mct


def refine_gpu_utilization(
    gpu_busy_time: float,
    *,
    application_runtime: float = 0.0,
    trace_runtime: float = 0.0,
    mpi_comm_time: float = 0.0,
    mpip_wall_fraction: float = 0.0,
    total_mpi_ranks: int = 1,
    cupti_process_count: int = 0,
    gpu_active_span: float = 0.0,
    nvidia_smi_util: float = 0.0,
    cpu_utilization: float = 0.0,
    profiling_phase_aligned: bool = True,
) -> Tuple[float, str]:
    """
    Compute δ-style GPU utilization in [0, 1] and a provenance tag.

    Combines CUPTI busy time, mpiP-aware hybrid fraction, optional rank coverage
    scaling, activity span, and nvidia-smi peak sampling.
    """
    busy = max(0.0, float(gpu_busy_time or 0.0))
    app_rt = max(float(application_runtime or 0.0), 0.0)
    trace_rt = max(float(trace_runtime or 0.0), 0.0)
    wall = app_rt if app_rt > 0 else trace_rt
    ranks = max(1, int(total_mpi_ranks or 1))
    procs = max(0, int(cupti_process_count or 0))
    span = max(0.0, float(gpu_active_span or 0.0))
    smi = max(0.0, min(1.0, float(nvidia_smi_util or 0.0)))

    if busy <= 1e-9 and span <= 1e-9 and smi <= 0.001:
        return 0.0, "no_gpu_metrics"

    mpi_sec = mpi_comm_seconds_for_gpu(wall, mpi_comm_time, mpip_wall_fraction)

    busy_adj = busy
    source = "cupt_time_ratio"
    if ranks > 1 and procs == 1 and busy > 0:
        # Single CUDA process in a multi-rank job — scale toward full-job GPU work.
        scale = min(float(ranks), max(1.0, wall / max(busy, 1e-6) * 0.15))
        busy_adj = min(busy * scale, wall * 0.95 if wall > 0 else busy * scale)
        source = "cupt_mpi_rank_scaled"
    elif ranks > 1 and procs > 1:
        source = "cupt_multi_process"

    candidates = []
    if wall > 0 and busy_adj > 0:
        candidates.append(("app_wall", min(1.0, busy_adj / wall)))
    if trace_rt > 0 and busy > 0:
        candidates.append(("trace_wall", min(1.0, busy / trace_rt)))
    if wall > 0 and span > 0:
        candidates.append(("cupt_span", min(1.0, span / wall)))
    denom = busy_adj + mpi_sec
    if denom > 1e-9:
        candidates.append(("hybrid_mpi_gpu", min(1.0, busy_adj / denom)))

    if not candidates:
        util = 0.0
    elif mpi_sec > 0 and wall > 0 and mpip_wall_fraction > 0.05:
        util = dict(candidates).get("hybrid_mpi_gpu", candidates[-1][1])
        source = f"{source}_hybrid"
    elif app_rt > 0 and (profiling_phase_aligned or ranks == 1):
        util = dict(candidates).get("app_wall", candidates[0][1])
        source = f"{source}_app_wall"
    elif ranks > 1 and wall > 0:
        util = max(dict(candidates).get("hybrid_mpi_gpu", 0.0), dict(candidates).get("app_wall", 0.0))
        source = f"{source}_mpi_app"
    else:
        util = candidates[0][1]

    if smi > 0.02:
        cup = max(1e-6, min(1.0, util))
        # Sub-ms runs: smi sampler often misses bursts; do not crush strong CUPTI (e.g. 1×48).
        if cup >= 0.35 and smi < 0.20:
            blended = cup
        elif wall > 0 and wall < 0.10 and cup > smi * 2.0:
            blended = cup
        else:
            blended = float((cup * smi) ** 0.5)
        util = max(util, blended) if cup >= 0.35 else blended
        source = f"{source}_nvsmi_peak"

    # When CUPTI sees only rank-0 bursts but mpiP shows moderate MPI, bound non-MPI compute
    # share so δ is not stuck near zero for legitimate GPU+MPI hybrid jobs.
    cpu_u = max(0.0, min(1.0, float(cpu_utilization or 0.0)))
    if (
        wall > 0
        and mpip_wall_fraction > 0.05
        and util < 0.12
        and busy_adj > 0
        and ranks > 1
    ):
        host_share = max(0.0, (1.0 - cpu_u) * (1.0 - mpip_wall_fraction))
        analytic = max(0.0, 1.0 - mpip_wall_fraction - host_share * 0.55)
        if analytic > util:
            util = min(0.92, analytic)
            source = f"{source}_hybrid_balance"

    util = float(max(0.0, min(1.0, util)))
    return util, source


def peak_nvidia_smi_util_from_profile_dir(profiling_dir: Path) -> float:
    """Peak GPU duty (0–1) from nvidia_smi_util_samples.csv, trimming init/finalize idle."""
    path = profiling_dir / "nvidia_smi_util_samples.csv"
    if not path.is_file():
        return 0.0
    try:
        vals: List[float] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                v = float(line.split(",")[0].strip())
            except ValueError:
                continue
            if 0.0 <= v <= 100.0:
                vals.append(v)
        if not vals:
            return 0.0
        n = len(vals)
        trimmed = vals[n // 10 : n - n // 10] if n >= 10 else vals
        return float(max(trimmed) / 100.0)
    except OSError:
        return 0.0
