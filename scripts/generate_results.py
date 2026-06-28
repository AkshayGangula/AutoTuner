#!/usr/bin/env python3
"""
Generate results from profiling data - Consolidated version.
This script extracts metrics from Nsight Systems profiling data and SLURM output files
to generate comprehensive reports and visualizations with 5 heuristic components.

Interpretation (no effect on scores or files—readability only):
  - The recommended configuration for raw performance is the fastest wall-clock time
    (runtime_sec); rankings and the summary label this explicitly.
  - Heuristic score (alpha–epsilon composite) summarizes efficiency and profiler signals;
    it can rank differently from runtime (e.g. GPU vs MPI tradeoffs, incomplete traces).
"""

import os
import sys
import argparse
import logging
import sqlite3
import re
import time
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import json
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# Import for visualization
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("ERROR: Visualization packages not available. Install pandas, matplotlib, seaborn.")
    sys.exit(1)

# Repo root (AutoTuner/) for package imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from autotuner.core.configuration_generator import detect_hardware_topology
from autotuner.core.metrics_extractor import NsightMetricsExtractor
from autotuner.core.phase1_subset_selector import backfill_phase1_throughput
from autotuner.app_registry.profiles import (
    LATEST_EXPERIMENT_DIR_PREFIXES,
    yaml_profile_key_for_experiment_dir_name,
)
from autotuner.app_registry.lulesh import (
    lulesh_throughput_z_per_s_from_mesh,
    parse_lulesh_mesh_iter_from_job_script,
)
from autotuner.app_registry.stdout_parsing import (
    infer_hybrid_vec_gflops_from_slurm_cli,
    parse_slurm_file_runtime_throughput as parse_slurm_output,
    parse_slurm_job_wall_clock_seconds,
)
from autotuner.utils.path_utils import (
    REPO_ROOT,
    experiment_collection_roots,
    experiment_path_roots,
    find_slurm_log_for_job,
)

# Split visualizations into dedicated modules under figures/
from autotuner.figures.dashboard import plot_auto_tuning_dashboard
from autotuner.figures.scores_heatmap import plot_scores_heatmap
from autotuner.figures.profiling_metrics import plot_profiling_metrics
from autotuner.figures.runtime_stacked_breakdown import plot_runtime_stacked_breakdown
from autotuner.figures.rank_agreement_plot import plot_rank_agreement

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_job_id_from_slurm_file(slurm_file: Path) -> Optional[str]:
    """Extract job ID from SLURM output filename (e.g., slurm-634790.out -> 634790)."""
    match = re.search(r'slurm-(\d+)\.(out|err)', slurm_file.name)
    return match.group(1) if match else None


def load_phase2_job_ids_from_file(path: Path) -> set:
    """Parse phase2_job_ids.json (list or dict with job_ids / phase2_job_ids / ids); never split JSON strings."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return {str(x) for x in raw}
    if isinstance(raw, dict):
        for key in ("job_ids", "phase2_job_ids", "ids"):
            v = raw.get(key)
            if isinstance(v, list):
                return {str(x) for x in v}
        return set()
    if isinstance(raw, str):
        return {raw}
    return set()


def gather_slurm_logs_by_job_id(experiment_dir: Path) -> Dict[str, Path]:
    """Map job id -> slurm path (prefer .out over .err); scans root and results/<job_id>/ (FAILED archives)."""
    by_job: Dict[str, Path] = {}

    def register(paths: List[Path]) -> None:
        for p in paths:
            if not p.is_file():
                continue
            jid = extract_job_id_from_slurm_file(p)
            if jid and jid not in by_job:
                by_job[jid] = p

    for root in experiment_path_roots(experiment_dir):
        for pattern in ("slurm-*.out", "slurm-*.err"):
            register(sorted(root.glob(pattern)))
            rr = root / "results"
            if rr.is_dir():
                register(sorted(rr.glob(f"*/{pattern}")))
    return by_job


def match_configuration_name_from_slurm_text(slurm_content: str, config_map: Dict[str, Any]) -> Optional[str]:
    """Match config label from SLURM/job-script text (Starting job / #SBATCH --job-name / substring)."""
    config_name = None
    m_job = re.search(
        r"(?:^|\n)Starting job:\s*lulesh_(\d+x\d+)_n\d+",
        slurm_content,
    ) or re.search(
        r"#SBATCH\s+--job-name=lulesh_(\d+x\d+)_n\d+",
        slurm_content,
    )
    if m_job:
        cand = m_job.group(1)
        if cand in config_map:
            config_name = cand

    if not config_name:
        m_app = re.search(
            r"(?:^|\n)Starting job:\s*(?:hybrid_vec|sparse_matrix|sparse|minimd)_(\d+x\d+)\b",
            slurm_content,
        ) or re.search(
            r"#SBATCH\s+--job-name=(?:hybrid_vec|sparse_matrix|sparse|minimd)_(\d+x\d+)\b",
            slurm_content,
            re.IGNORECASE,
        )
        if m_app:
            cand = m_app.group(1)
            if cand in config_map:
                config_name = cand

    if not config_name:
        for name in sorted(config_map.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, slurm_content):
                config_name = name
                break

    if not config_name:
        for name in sorted(config_map.keys(), key=len, reverse=True):
            if name in slurm_content:
                config_name = name
                break

    return config_name


def phase2_config_from_job_script(experiment_dir: Path, job_id: str, config_map: Dict[str, Any]) -> Optional[str]:
    """Resolve config from archived results/<job_id>/job_script.slurm using the same rules as SLURM logs."""
    jid = str(job_id)
    for root in experiment_path_roots(experiment_dir):
        js = root / "results" / jid / "job_script.slurm"
        if not js.is_file():
            continue
        try:
            text = js.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cn = match_configuration_name_from_slurm_text(text, config_map)
        if cn:
            return cn
    return None


def resolve_profiling_dir(experiment_dir: Path, job_id: str) -> Path:
    """
    Prefer results/<job_id>/profiling/ (where result_collector archives Nsight output).
    Fall back to profiling/<job_id>/ (live experiment layout).
    Tries logical and backing-fs roots (e.g. /home vs /mmfs1/home).
    """
    jid = str(job_id)
    for root in experiment_path_roots(experiment_dir):
        for d in (
            root / "results" / jid / "profiling",
            root / "profiling" / jid,
        ):
            if d.is_dir():
                return d
    roots = experiment_path_roots(experiment_dir)
    return roots[0] / "profiling" / jid


def _read_nvidia_smi_samples(profiling_dir: Path) -> List[float]:
    """Raw nvidia-smi utilization.gpu samples (0–100) from Run 2 profiling."""
    path = profiling_dir / "nvidia_smi_util_samples.csv"
    if not path.is_file():
        return []
    try:
        vals: List[float] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split(",")[0].strip()
            try:
                v = float(tok)
            except ValueError:
                continue
            if 0.0 <= v <= 100.0:
                vals.append(v)
        return vals
    except OSError:
        return []


def mean_nvidia_smi_util_from_profile_dir(profiling_dir: Path) -> float:
    """
    Average `nvidia-smi utilization.gpu` samples (0–100) written during Run 2 profiling
    to nvidia_smi_util_samples.csv. Returns fraction in [0, 1], or 0.0 if missing.
    """
    vals = _read_nvidia_smi_samples(profiling_dir)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals) / 100.0)


def peak_nvidia_smi_util_from_profile_dir(profiling_dir: Path) -> float:
    """Peak GPU duty during profiling (delegates to core.profiling_refinement)."""
    from autotuner.core.profiling_refinement import peak_nvidia_smi_util_from_profile_dir as _peak

    return _peak(profiling_dir)


def infer_slurm_num_nodes(experiment_dir: Path) -> int:
    """
    Read #SBATCH --nodes=N from a generated job script (fallback when a config has no
    per-job ``num_nodes`` in generated_configurations.json, e.g. hybrid_vec runs).
    """
    scripts_dir = experiment_dir / "scripts"
    if not scripts_dir.is_dir():
        return 1
    for slurm in sorted(scripts_dir.glob("*.slurm")):
        try:
            text = slurm.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"^#SBATCH\s+--nodes=(\d+)", text, re.MULTILINE)
        if m:
            n = int(m.group(1))
            return max(1, n)
    return 1


def build_experiment_provenance(
    experiment_dir: Path,
    *,
    enable_locality: bool = False,
    cpu_only: bool = True,
    adaptive_weights: bool = False,
) -> Dict[str, Any]:
    """
    Reproducibility / provenance metadata for research artifacts. Embedded in comprehensive_results/results.json.
    """
    repo_root = Path(__file__).resolve().parent
    prov: Dict[str, Any] = {
        "schema_version": "1.0",
        "results_json_layout": "wrapped_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "experiment_dir": str(experiment_dir.resolve()),
        "python_version": platform.python_version(),
        "os_summary": platform.platform(),
        "heuristic_model": {
            "axes": ["alpha_comm", "beta_thread", "gamma_locality", "delta_gpu", "epsilon_openmp"],
            "adaptive_weights": adaptive_weights,
            "weight_mode": "static_baseline_with_profile_layout_redistribution",
            "provenance_fields": [
                "alpha_source",
                "beta_source",
                "gamma_locality",
                "locality_source",
                "delta_gpu_source",
                "delta_profile_coverage",
                "profiling_phase_aligned",
            ],
            "enable_locality": enable_locality,
            "cpu_only_mode": cpu_only,
        },
    }
    for pkg in ("numpy", "pandas"):
        try:
            __import__(pkg)
            prov[f"{pkg}_version"] = getattr(sys.modules[pkg], "__version__", "unknown")
        except Exception:
            pass
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            prov["anti_gravity_git_commit"] = cp.stdout.strip()
    except Exception:
        pass
    for name in ("generated_configurations.json", "phase1_results.json", "phase2_job_ids.json"):
        prov[f"has_{name.replace('.json', '')}"] = (experiment_dir / name).is_file()
    return prov


def _max_gpu_utilization_metrics(all_metrics: List[Dict[str, Any]]) -> float:
    """Largest Nsight-derived ``gpu_utilization`` across collected rows (0–1 scale)."""
    m = 0.0
    for row in all_metrics:
        try:
            m = max(m, float(row.get("gpu_utilization") or 0.0))
        except (TypeError, ValueError):
            continue
    return m


def apply_cpu_only_when_gpu_absent(
    experiment_dir: Path,
    all_metrics: List[Dict[str, Any]],
    cpu_only: bool,
    gpu_util_threshold: float = 0.02,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    If GPU-mode scoring was requested but every row shows negligible GPU utilization, switch to
    CPU-only weights and re-collect metrics (matches ``master_tuner``'s scorer fallback).

    Returns ``(effective_cpu_only, metrics)``.
    """
    if not all_metrics or cpu_only:
        return cpu_only, all_metrics
    max_gpu = _max_gpu_utilization_metrics(all_metrics)
    if max_gpu > gpu_util_threshold:
        return cpu_only, all_metrics
    logger.warning(
        "Negligible GPU utilization across configs (max=%.4f). Using CPU-only heuristic weights "
        "(δ redistributed to α/β) and re-collecting metrics. Rebuild/run a working GPU binary on "
        "compute nodes for full δ scoring.",
        max_gpu,
    )
    return True, collect_experiment_data(experiment_dir, cpu_only=True)


def collect_experiment_data(experiment_dir: Path, cpu_only: bool = True) -> List[Dict[str, Any]]:
    """Collect all metrics from an experiment directory."""
    all_metrics = []
    
    logger.info(f"📂 Experiment directory: {experiment_dir}")
    
    # Read generated configurations
    config_file = experiment_dir / "generated_configurations.json"
    if not config_file.exists():
        logger.error(f"❌ Configuration file not found: {config_file}")
        logger.error(f"   Make sure the experiment directory contains generated_configurations.json")
        return []
    
    try:
        configs = json.loads(config_file.read_text())
        logger.info(f"✅ Loaded {len(configs)} configurations from {config_file.name}")
    except Exception as e:
        logger.error(f"❌ Failed to parse configuration file: {e}")
        return []
    
    # Check for duplicate config names
    config_names = [cfg['name'] for cfg in configs]
    if len(config_names) != len(set(config_names)):
        duplicates = [name for name in config_names if config_names.count(name) > 1]
        logger.warning(f"⚠️  Found duplicate config names: {set(duplicates)}. Using last occurrence.")
    
    # Phase 1 filtering: If phase1_results.json exists, ONLY output configs selected in Phase 1.
    # This prevents rejected configs (which have tiny Phase 1 runtimes) from falsely winning Phase 2 ranking.
    phase1_file = experiment_dir / "phase1_results.json"
    if phase1_file.exists():
        try:
            p1_data = json.loads(phase1_file.read_text())
            if "phase1_selected" in p1_data:
                selected_set = set(p1_data["phase1_selected"])
                original_count = len(configs)
                configs = [c for c in configs if c['name'] in selected_set]
                logger.info(f"✅ Phase-1 filtering: restricting analysis to {len(configs)} selected configurations (dropped {original_count - len(configs)} rejected configs)")
        except Exception as e:
            logger.warning(f"Could not apply Phase-1 filtering: {e}")

    # Use last occurrence for duplicates (or could use first - document the choice)
    config_map = {cfg['name']: cfg for cfg in configs}

    num_nodes = infer_slurm_num_nodes(experiment_dir)
    if num_nodes > 1:
        logger.info(
            f"✅ Inferred {num_nodes} SLURM nodes from scripts/*.slurm — "
            "heuristic masks (α–ε) use total_mpi_ranks = nodes × mpi_ranks_per_node"
        )

    slurm_by_job = gather_slurm_logs_by_job_id(experiment_dir)
    slurm_files = sorted(
        slurm_by_job.values(),
        key=lambda f: int(extract_job_id_from_slurm_file(f) or 0),
    )

    # Two-phase mode: only use Phase-2 job outputs (full profiling) when phase2_job_ids.json exists
    phase2_job_ids_file = experiment_dir / "phase2_job_ids.json"
    phase2_job_ids: Optional[Set[str]] = None
    if phase2_job_ids_file.exists():
        try:
            phase2_job_ids = load_phase2_job_ids_from_file(phase2_job_ids_file)
            deep_ids_file = experiment_dir / "deep_profile_job_ids.json"
            if deep_ids_file.exists():
                try:
                    extra = json.loads(deep_ids_file.read_text())
                    if isinstance(extra, list):
                        phase2_job_ids |= set(str(j) for j in extra)
                    logger.info(
                        f"✅ Merged deep_profile_job_ids.json — SLURM filter includes "
                        f"{len(phase2_job_ids)} job id(s)"
                    )
                except Exception as e:
                    logger.warning(f"Could not read deep_profile_job_ids.json: {e}")
            before_ct = len(slurm_files)
            slurm_files = [f for f in slurm_files if extract_job_id_from_slurm_file(f) in phase2_job_ids]
            logger.info(
                f"✅ Two-phase mode: discovered {len(slurm_files)} SLURM log file(s) matching "
                f"{len(phase2_job_ids)} Phase-2 job id(s)"
            )
            if before_ct > 0 and len(slurm_files) == 0:
                logger.warning(
                    "Phase-2 SLURM filter: no logs match phase2_job_ids.json ids %s. "
                    "Found slurm-* for other job id(s) %s (often Phase-1 ids for the same config names). "
                    "Phase-2 jobs must succeed to write matching slurm-<phase2_id>.* — FAILED Phase-2 leaves only Phase-1 logs.",
                    sorted(phase2_job_ids),
                    sorted(slurm_by_job.keys()),
                )
        except Exception as e:
            logger.warning(f"Could not read phase2_job_ids.json: {e}")

    # Build work items: SLURM-backed rows plus Phase-2 jobs with no slurm-* on disk (FAILED early —
    # resolve config from results/<job_id>/job_script.slurm so profiling/LIKWID can still be merged).
    work_items: List[Tuple[str, str, Optional[Path]]] = []
    seen_job_ids: Set[str] = set()

    for slurm_file in slurm_files:
        job_id = extract_job_id_from_slurm_file(slurm_file)
        if not job_id:
            continue
        slurm_content = slurm_file.read_text(encoding="utf-8", errors="replace")
        config_name = match_configuration_name_from_slurm_text(slurm_content, config_map)
        if not config_name:
            logger.warning(f"Could not match job {job_id} to a configuration from SLURM text")
            continue
        work_items.append((job_id, config_name, slurm_file))
        seen_job_ids.add(job_id)

    if phase2_job_ids:
        for jid in sorted(phase2_job_ids, key=lambda x: int(x)):
            if jid in seen_job_ids:
                continue
            cn_guess = phase2_config_from_job_script(experiment_dir, jid, config_map)
            if not cn_guess or cn_guess not in config_map:
                logger.warning(
                    "Phase-2 job %s: could not resolve configuration from results/%s/job_script.slurm — skipping",
                    jid,
                    jid,
                )
                continue
            sp = find_slurm_log_for_job(experiment_dir, jid)
            if sp is None:
                logger.warning(
                    "Phase-2 job %s (%s): no slurm-*.out/.err found — continuing via job_script "
                    "(metrics need profiling/sqlite or mpiP under results/%s/)",
                    jid,
                    cn_guess,
                    jid,
                )
            work_items.append((jid, cn_guess, sp))
            seen_job_ids.add(jid)

    if not work_items:
        logger.error(f"❌ No Phase-2 configurations to process under {experiment_dir}")
        if not slurm_files:
            logger.error(
                "   No SLURM logs found; with phase2_job_ids.json, ensure results/<job_id>/job_script.slurm exists."
            )
        if phase2_job_ids_file.exists():
            try:
                exp = load_phase2_job_ids_from_file(phase2_job_ids_file)
                if exp:
                    logger.error(f"   Phase-2 job id(s): {sorted(exp)} — inspect results/<id>/ and sacct.")
            except Exception:
                pass
        return []

    logger.info(
        f"✅ Processing {len(work_items)} Phase-2 configuration(s) "
        f"({sum(1 for _, _, p in work_items if p is not None and p.is_file())} with SLURM logs)"
    )

    for job_id, config_name, slurm_path in work_items:
        config = config_map[config_name]
        discovered = find_slurm_log_for_job(experiment_dir, job_id)
        if discovered is not None:
            slurm_path = discovered

        slurm_content = ""
        if slurm_path is not None and slurm_path.is_file():
            slurm_content = slurm_path.read_text(encoding="utf-8", errors="replace")
            alt = slurm_path.with_suffix(".err" if slurm_path.suffix.lower() == ".out" else ".out")
            if alt.is_file() and alt.resolve() != slurm_path.resolve():
                slurm_content = slurm_content + "\n" + alt.read_text(encoding="utf-8", errors="replace")
        logger.info(f"Processing {config_name} (Job {job_id})...")

        # Parse SLURM output when present (try sibling .out/.err — some sites route app output to stderr only)
        if slurm_path is not None and slurm_path.is_file():
            runtime, throughput = parse_slurm_output(slurm_path)
            if runtime <= 0.0:
                altp = slurm_path.with_suffix(".err" if slurm_path.suffix.lower() == ".out" else ".out")
                if altp.is_file() and altp.resolve() != slurm_path.resolve():
                    r2, t2 = parse_slurm_output(altp)
                    if r2 > 0:
                        runtime, throughput = r2, t2 if t2 > 0 else throughput
        else:
            runtime, throughput = 0.0, 0.0
        if runtime == 0.0:
            # Fallback: try Run 1 output (multi-rank light application) if collected
            run1_name = f"run1_{job_id}.out"
            for root in experiment_path_roots(experiment_dir):
                for run1_path in (root / "results" / job_id / run1_name, root / run1_name):
                    if run1_path.exists():
                        r1_rt, r1_tp = parse_slurm_output(run1_path)
                        if r1_rt > 0:
                            runtime, throughput = r1_rt, r1_tp
                            logger.info(
                                f"  Runtime/throughput from Run 1 output: {runtime:.6f}s, {throughput:.2f} "
                                f"(GFLOPS or z/s)"
                            )
                            break
                if runtime > 0:
                    break
            if runtime == 0.0:
                logger.debug(
                    "  No parsable app timing in SLURM/run1; will try Nsight trace and SLURM Start/End wall clock."
                )
        else:
            logger.info(
                f"  Runtime: {runtime:.6f}s, Throughput: {throughput:.2f} "
                f"(GFLOPS for hybrid_vec; z/s for LULESH)"
            )
        
        # Extract REAL metrics using NsightMetricsExtractor (not proxies)
        profiling_dir = resolve_profiling_dir(experiment_dir, job_id)
        
        # Initialize real metrics (will be extracted from profiling data)
        mpi_comm_time = 0.0
        cpu_utilization = 0.0
        openmp_work_efficiency = 0.0
        openmp_instrumented = False  # Legacy column; ε comes from trace-derived openmp_work_efficiency
        gpu_utilization = 0.0
        gpu_utilization_source = "no_gpu_metrics"
        numa_efficiency = 0.0
        locality_source = "missing"
        thread_stall_time = 0.0
        memory_bandwidth = 0.0  # Nsight/LIKWID-derived GB/s (aggregate across profiled ranks)
        profiling_runtime_used = False  # True when wall time taken from Nsight trace (not app printf)
        wall_runtime_used = False  # True when using Start/End date delta from job script (last resort)
        profile_total_runtime = 0.0  # Nsight trace duration; use for α comm_fraction denominator
        mpip_app_time = 0.0  # mpiP AppTime (max rank); preferred α denominator when available
        mpip_max_mpi_wall_fraction = 0.0  # mpiP max_i(MPI_i/(App_i+MPI_i)); preferred α comm fraction when > 0
        gpu_busy_time = 0.0  # CUPTI kernel+memcpy+sync seconds; for wall-aligned δ
        nvidia_smi_gpu_util = 0.0  # driver-reported utilization sample mean [0,1]
        gpu_utilization_cupti = 0.0  # CUPTI time-ratio util after optional wall alignment (before nvidia-smi blend)
        is_two_run = False  # multi-rank per node + single rank-0 profile → wall clock may not match trace phase
        profiling_phase_aligned = True
        num_profiles = 0
        
        
        if profiling_dir.exists():
            # Look for .sqlite files (converted from .nsys-rep)
            all_sqlite_files = list(profiling_dir.glob("*.sqlite"))
            
            # Filter out empty files (0 bytes) upfront
            sqlite_files = [f for f in all_sqlite_files if f.stat().st_size > 0]
            
            # Prioritize profile.sqlite (single-rank) over profile_rank_*.sqlite (multi-rank)
            # This handles cases where both exist but profile_rank_0.sqlite is empty
            profile_main = profiling_dir / "profile.sqlite"
            if profile_main.exists() and profile_main.stat().st_size > 0:
                # Use profile.sqlite, ignore profile_rank_* files
                sqlite_files = [profile_main]
            else:
                # Use profile_rank_* files (multi-rank case; deep profiling may produce several ranks)
                sqlite_files = sorted(
                    (f for f in sqlite_files if f.name.startswith("profile_rank_")),
                    key=lambda p: p.name,
                )
            
            num_profiles = len(sqlite_files)
            
            # Also check for .nsys-rep files to detect incomplete conversion
            nsys_files = list(profiling_dir.glob("*.nsys-rep"))
            
            if num_profiles == 0:
                if nsys_files:
                    logger.warning(f"  ⚠️  Found {len(nsys_files)} .nsys-rep files but no .sqlite files.")
                    logger.warning(
                        "     Convert to .sqlite for Nsight-based β (thread) and δ (GPU); "
                        "α may still come from mpiP, γ from LIKWID (when present)."
                    )
                else:
                    logger.warning(f"  ⚠️  No Nsight SQLite (or .nsys-rep) in {profiling_dir}")
                    logger.warning(
                        "     Trace-based β/δ (and Nsight-α) unavailable for this config; "
                        "α may still come from mpiP, γ from LIKWID; ε uses cross-config normalization if needed."
                    )
            elif len(nsys_files) > num_profiles:
                # Some .nsys-rep files haven't been converted to .sqlite
                # (Multi-rank runs: one full-job .nsys-rep/sqlite for all ranks; extras may be from an older run.)
                missing_sqlite = len(nsys_files) - num_profiles
                logger.warning(f"  ⚠️  Incomplete SQLite conversion: {num_profiles} .sqlite files but {len(nsys_files)} .nsys-rep files")
                logger.warning(f"     {missing_sqlite} profile(s) missing SQLite conversion. Metrics may be incomplete.")
            
            if sqlite_files:
                logger.info(f"  Found {num_profiles} profile file(s) - extracting REAL metrics...")
                if num_profiles == 1 and config.get('mpi_ranks_per_node', 1) > 1:
                    logger.info(
                        "  (Multi-rank: one full-job Nsight sqlite — all ranks in session; α from Nsight/MPI tables unless .mpiP present.)"
                    )
                elif num_profiles > 1:
                    logger.info(
                        f"  (Multi-rank: aggregating {num_profiles} Nsight SQLite file(s).)"
                    )
                
                # Extract real metrics from all profile files (aggregate for multi-rank)
                all_extracted_metrics = []
                for sqlite_file in sqlite_files:
                    try:
                        # Validate SQLite file is readable
                        if not sqlite_file.exists() or sqlite_file.stat().st_size == 0:
                            logger.warning(f"  SQLite file is missing or empty: {sqlite_file}")
                            continue
                        
                        with NsightMetricsExtractor(sqlite_file) as extractor:
                            extracted = extractor.extract_all_metrics(backend_prefix="MPI_")
                            if extracted is not None:
                                all_extracted_metrics.append(extracted)
                            else:
                                logger.warning(f"  Extracted metrics is None from {sqlite_file}")
                    except sqlite3.DatabaseError as e:
                        logger.warning(f"  SQLite database error (file may be corrupted): {sqlite_file}: {e}")
                    except Exception as e:
                        logger.warning(f"  Failed to extract real metrics from {sqlite_file}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())

                # Aggregate real metrics from profile sqlite(s) (one file = full-job trace for multi-rank)
                total_runtime = 0.0  # Initialize before use
                if all_extracted_metrics:
                    # Filter out None/NaN values before aggregation
                    def safe_value(val, default=0.0):
                        """Return value if valid, else default"""
                        if val is None:
                            return default
                        try:
                            val_float = float(val)
                            if np.isnan(val_float) or np.isinf(val_float):
                                return default
                            return val_float
                        except (TypeError, ValueError):
                            return default
                    
                    if len(all_extracted_metrics) > 1:
                        # Multiple profiled ranks (deep spread: ranks 0, N/2, N-1).
                        # MPI comm time: max (mpiP is per-job, duplicated per rank DB).
                        # Runtime: max wall-clock across traced ranks.
                        valid_metrics = []
                        for m in all_extracted_metrics:
                            if m is not None:
                                valid_metrics.append(m)
                        
                        if valid_metrics:
                            # mpiP / MPI times are per-job (same report for all ranks); do not sum duplicate extracts.
                            mpi_comm_time = max(safe_value(m.mpi_comm_time, 0.0) for m in valid_metrics)
                            # Runtime: use max (wall-clock time)
                            total_runtime = max(safe_value(m.total_runtime, 0.0) for m in valid_metrics)
                            # mpiP AppTime: use max rank app time across profiled files
                            mpip_app_time = max(safe_value(getattr(m, 'mpip_app_time', 0.0), 0.0) for m in valid_metrics)
                            # CPU utilization: average (per-rank utilization)
                            cpu_vals = [safe_value(m.cpu_utilization, 0.0) for m in valid_metrics]
                            cpu_utilization = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0
                            # Thread stall: average across traced ranks (per-rank view of β)
                            stall_vals = [safe_value(m.thread_stall_time, 0.0) for m in valid_metrics]
                            thread_stall_time = (
                                sum(stall_vals) / len(stall_vals) if stall_vals else 0.0
                            )
                            # Average OpenMP/GPU/NUMA metrics across ranks
                            openmp_vals = [safe_value(m.openmp_work_efficiency, 0.0) for m in valid_metrics]
                            openmp_work_efficiency = sum(openmp_vals) / len(openmp_vals) if openmp_vals else 0.0
                            openmp_instrumented = any(getattr(m, 'openmp_instrumented', False) for m in valid_metrics)
                            gpu_vals = [safe_value(m.gpu_utilization, 0.0) for m in valid_metrics]
                            gpu_utilization = sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0
                            wall_frac_vals = [
                                safe_value(getattr(m, 'mpip_max_mpi_wall_fraction', 0.0), 0.0)
                                for m in valid_metrics
                            ]
                            mpip_max_mpi_wall_fraction = (
                                max(wall_frac_vals) if wall_frac_vals else 0.0
                            )
                            busy_vals = [safe_value(getattr(m, 'gpu_busy_time', 0.0), 0.0) for m in valid_metrics]
                            gpu_busy_time = sum(busy_vals) / len(busy_vals) if busy_vals else 0.0
                            numa_vals = [safe_value(m.numa_efficiency, 0.0) for m in valid_metrics]
                            numa_efficiency = sum(numa_vals) / len(numa_vals) if numa_vals else 0.0
                            bw_vals = [safe_value(getattr(m, "memory_bandwidth", 0.0), 0.0) for m in valid_metrics]
                            memory_bandwidth = (
                                sum(bw_vals) / len(bw_vals) if bw_vals else 0.0
                            )
                            locality_sources = [getattr(m, 'locality_source', 'missing') for m in valid_metrics]
                            if "dram_local_remote" in locality_sources:
                                locality_source = "dram_local_remote"
                            elif "numa_meminfo" in locality_sources:
                                locality_source = "numa_meminfo"
                            elif "l3_estimate" in locality_sources:
                                locality_source = "l3_estimate"
                            elif "profiling" in locality_sources:
                                locality_source = "profiling"
                            else:
                                locality_source = "missing"
                            
                            if mpi_comm_time > total_runtime and total_runtime > 0:
                                logger.info(
                                    "  MPI comm %.3fs > application trace window %.3fs — capping for α (multi-rank / mpiP vs wall).",
                                    mpi_comm_time,
                                    total_runtime,
                                )
                                mpi_comm_time = total_runtime
                            profile_total_runtime = total_runtime  # For α: comm_fraction = mpi_comm_time / profile_total_runtime
                            # Validate: CPU utilization should be in [0, 1]
                            cpu_utilization = max(0.0, min(1.0, cpu_utilization))
                            # For pure MPI configs (omp_threads == 1), OpenMP efficiency is N/A
                            if config['omp_threads_per_rank'] == 1:
                                openmp_work_efficiency = 0.0
                            else:
                                # Trace-derived ε; if missing, downstream applies relative-runtime scaling
                                if openmp_work_efficiency is None or openmp_work_efficiency <= 0:
                                    openmp_work_efficiency = None
                                else:
                                    openmp_work_efficiency = max(0.0, min(1.0, openmp_work_efficiency))
                            gpu_utilization = max(0.0, min(1.0, gpu_utilization))
                            numa_efficiency = max(0.0, min(1.0, numa_efficiency))
                        else:
                            logger.warning(f"  ⚠️  No valid metrics found in extracted data")
                            total_runtime = 0.0  # Reset if no valid metrics
                    else:
                        # Single rank
                        m = all_extracted_metrics[0]
                        if m is not None:
                            mpi_comm_time = safe_value(m.mpi_comm_time, 0.0)
                            total_runtime = safe_value(m.total_runtime, 0.0)
                            profile_total_runtime = total_runtime
                            mpip_app_time = safe_value(getattr(m, 'mpip_app_time', 0.0), 0.0)
                            cpu_utilization = safe_value(m.cpu_utilization, 0.0)
                            thread_stall_time = safe_value(m.thread_stall_time, 0.0)
                            openmp_work_efficiency = safe_value(m.openmp_work_efficiency, 0.0)
                            openmp_instrumented = getattr(m, 'openmp_instrumented', False)
                            gpu_utilization = safe_value(m.gpu_utilization, 0.0)
                            mpip_max_mpi_wall_fraction = safe_value(
                                getattr(m, 'mpip_max_mpi_wall_fraction', 0.0), 0.0
                            )
                            gpu_busy_time = safe_value(getattr(m, 'gpu_busy_time', 0.0), 0.0)
                            numa_efficiency = safe_value(m.numa_efficiency, 0.0)
                            locality_source = getattr(m, 'locality_source', 'missing')
                            memory_bandwidth = safe_value(getattr(m, "memory_bandwidth", 0.0), 0.0)
                            
                            if mpi_comm_time > total_runtime and total_runtime > 0:
                                logger.info(
                                    "  MPI comm %.3fs > application trace window %.3fs — capping for α.",
                                    mpi_comm_time,
                                    total_runtime,
                                )
                                mpi_comm_time = total_runtime
                            
                            # Validate: metrics should be in valid ranges
                            cpu_utilization = max(0.0, min(1.0, cpu_utilization))
                            # For pure MPI configs (omp_threads == 1), OpenMP efficiency is N/A
                            if config['omp_threads_per_rank'] == 1:
                                openmp_work_efficiency = 0.0
                            else:
                                # Trace-derived ε; if missing, downstream applies relative-runtime scaling
                                if openmp_work_efficiency is None or openmp_work_efficiency <= 0:
                                    openmp_work_efficiency = None
                                else:
                                    openmp_work_efficiency = max(0.0, min(1.0, openmp_work_efficiency))
                            gpu_utilization = max(0.0, min(1.0, gpu_utilization))
                            numa_efficiency = max(0.0, min(1.0, numa_efficiency))
                        else:
                            logger.warning(f"  ⚠️  Extracted metric is None")
                            total_runtime = 0.0  # Reset if metric is None
                    
                    # Use runtime from profiling only if SLURM/Run1 runtime is missing (0.0).
                    # Application output (runtime) is the ground truth when available; Nsight runtime includes overhead.
                    # Multi-rank with one full-job profile: prefer SLURM/Run1 wall when parsed (trace wall can include Nsight overhead).
                    # If still 0 (e.g. LULESH -q with no timing in slurm-*.out, run1_*.out not in experiment dir),
                    # use Nsight trace duration so configs are not dropped from comprehensive_results.
                    is_two_run = (
                        num_profiles == 1 and int(config.get('mpi_ranks_per_node', 1) or 1) > 1
                    )
                    profiling_phase_aligned = not is_two_run
                    if runtime == 0.0 and total_runtime > 0:
                        runtime = total_runtime
                        profiling_runtime_used = True
                        if is_two_run:
                            logger.info(
                                "  Runtime from Nsight trace (multi-rank full-job profile; "
                                "no parsable wall time in SLURM/Run1 output)."
                            )
                    else:
                        profiling_runtime_used = False
                    
                    gpu_utilization_cupti = float(gpu_utilization)
                    nvidia_smi_mean = 0.0
                    nvidia_smi_peak = 0.0
                    if profiling_dir.exists():
                        nvidia_smi_mean = mean_nvidia_smi_util_from_profile_dir(profiling_dir)
                        nvidia_smi_peak = peak_nvidia_smi_util_from_profile_dir(profiling_dir)
                    nvidia_smi_gpu_util = max(nvidia_smi_mean, nvidia_smi_peak)

                    from autotuner.core.profiling_refinement import refine_gpu_utilization

                    mpi_ranks_cfg = int(config.get("mpi_ranks_per_node", 1) or 1) * int(
                        config.get("num_nodes", 1) or 1
                    )
                    cupti_procs = 0
                    gpu_span = 0.0
                    if all_extracted_metrics:
                        cupti_procs = max(
                            int(getattr(m, "cupti_process_count", 0) or 0)
                            for m in all_extracted_metrics
                            if m is not None
                        )
                        gpu_span = max(
                            float(getattr(m, "gpu_active_span", 0.0) or 0.0)
                            for m in all_extracted_metrics
                            if m is not None
                        )
                    gu_refined, gu_src = refine_gpu_utilization(
                        gpu_busy_time,
                        application_runtime=runtime,
                        trace_runtime=profile_total_runtime,
                        mpi_comm_time=mpi_comm_time,
                        mpip_wall_fraction=mpip_max_mpi_wall_fraction,
                        total_mpi_ranks=max(1, mpi_ranks_cfg),
                        cupti_process_count=cupti_procs,
                        gpu_active_span=gpu_span,
                        nvidia_smi_util=nvidia_smi_peak,
                        cpu_utilization=cpu_utilization,
                        profiling_phase_aligned=profiling_phase_aligned,
                    )
                    if gu_refined > 0 or gpu_busy_time > 0:
                        if abs(gu_refined - gpu_utilization) > 0.02:
                            logger.info(
                                f"  GPU util refined: {gpu_utilization:.3f} → {gu_refined:.3f} ({gu_src})"
                            )
                        gpu_utilization = gu_refined
                    gpu_utilization_source = gu_src
                    
                    # Ensure all metrics are floats (not None) before logging
                    safe_mpi_comm = safe_value(mpi_comm_time, 0.0)
                    safe_cpu_util = safe_value(cpu_utilization, 0.0)
                    safe_openmp_eff = safe_value(openmp_work_efficiency, 0.0)
                    safe_gpu_util = safe_value(gpu_utilization, 0.0)
                    
                    eps_label = "ε (OpenMP)"
                    logger.info(f"  ✅ Real metrics: MPI comm={safe_mpi_comm:.3f}s, CPU util={safe_cpu_util:.3f}, "
                              f"OpenMP eff={safe_openmp_eff:.3f} ({eps_label}), GPU util={safe_gpu_util:.3f}")
                else:
                    # No extracted metrics - profiling failed
                    total_runtime = 0.0
                    profiling_runtime_used = False
                    logger.warning(f"  ⚠️  Could not extract real metrics from profiling files")
                    if profiling_dir.exists():
                        nvidia_smi_gpu_util = mean_nvidia_smi_util_from_profile_dir(profiling_dir)

        # Prefer mpiP for α when Run 1a reports exist (PMPI timing beats Nsight OSRT on MVAPICH).
        mpip_data = NsightMetricsExtractor.load_mpip_comm_metrics_from_experiment(
            experiment_dir.resolve(), str(job_id)
        )
        mpip_frac = float(mpip_data.get("mpip_max_mpi_wall_fraction", 0.0) or 0.0)
        mpip_app = float(mpip_data.get("max_rank_app_time", 0.0) or 0.0)
        mpip_tc = float(mpip_data.get("total_comm_time", 0.0) or 0.0)
        if mpip_frac > 0.001 or mpip_app > 0.001:
            if mpip_tc > 0.0:
                mpi_comm_time = mpip_tc
            mpip_app_time = mpip_app
            mpip_max_mpi_wall_fraction = mpip_frac
            cap_rt = runtime if runtime > 0 else profile_total_runtime
            if cap_rt > 0 and mpi_comm_time > cap_rt:
                logger.info(
                    "  mpiP MPI time %.4fs > application runtime %.4fs — capping for α.",
                    mpi_comm_time,
                    cap_rt,
                )
                mpi_comm_time = min(mpi_comm_time, cap_rt)
            logger.info(
                f"  α from mpiP: MPI comm={mpi_comm_time:.4f}s, "
                f"mpip_app={mpip_app_time:.4f}s, wall_frac={mpip_max_mpi_wall_fraction:.3f}"
            )
        elif mpi_comm_time <= 0.0 and mpip_tc > 0.0:
            mpi_comm_time = mpip_tc
            mpip_app_time = mpip_app
            mpip_max_mpi_wall_fraction = mpip_frac
            logger.info(
                f"  α from mpiP (no Nsight MPI rows): MPI comm={mpi_comm_time:.4f}s, "
                f"wall_frac={mpip_max_mpi_wall_fraction:.3f}"
            )

        # For pure MPI configs (omp_threads == 1), OpenMP efficiency is N/A (set to 0.0)
        if config['omp_threads_per_rank'] == 1:
            openmp_work_efficiency = 0.0
        else:
            # Ensure openmp_work_efficiency is a float (not None) for storage
            if openmp_work_efficiency is None:
                openmp_work_efficiency = 0.0
        
        # Ensure all metrics are floats (not None) before storing
        # Convert None to 0.0 for all metrics to avoid formatting errors
        if mpi_comm_time is None:
            mpi_comm_time = 0.0
        if cpu_utilization is None:
            cpu_utilization = 0.0
        if gpu_utilization is None:
            gpu_utilization = 0.0
        if numa_efficiency is None:
            numa_efficiency = 0.0
        if thread_stall_time is None:
            thread_stall_time = 0.0
        # LIKWID is independent of Nsight; use it for locality when we have no NUMA from profiling (e.g. multi-rank without Run 2)
        if numa_efficiency == 0.0:
            from autotuner.core.likwid_profiler import get_locality_for_job
            likwid_eff, likwid_src = get_locality_for_job(experiment_dir, job_id)
            if likwid_eff > 0:
                numa_efficiency = likwid_eff
                locality_source = likwid_src
                logger.info(f"  Locality (γ) from LIKWID: {numa_efficiency:.3f}")

        try:
            from autotuner.core.likwid_profiler import resolve_memory_bandwidth_for_job
            memory_bandwidth, bw_src = resolve_memory_bandwidth_for_job(
                experiment_dir, job_id, memory_bandwidth
            )
            if memory_bandwidth > 0:
                logger.info(f"  Memory BW for γ: {memory_bandwidth:.3f} GB/s ({bw_src})")
        except (ImportError, Exception):
            pass

        # Last-resort runtime: AutoTuner job scripts echo Start/End from `date` when the app prints nothing.
        if runtime <= 0.0:
            wall_s = parse_slurm_job_wall_clock_seconds(slurm_content)
            if wall_s > 0.0:
                runtime = wall_s
                wall_runtime_used = True
                profiling_runtime_used = False
                logger.warning(
                    "  ⚠️  Using SLURM job Start/End wall clock (%.3fs) as runtime — no application timing "
                    "(e.g. Exec format error or crash before HYBRID_VEC_GPU/LULESH output). "
                    "Wall time includes LIKWID/Nsight phases, not pure kernel time. Rebuild on the target nodes for real metrics.",
                    wall_s,
                )

        if wall_runtime_used and throughput <= 0.0:
            est_tp = infer_hybrid_vec_gflops_from_slurm_cli(slurm_content, runtime)
            if est_tp <= 0.0:
                for root in experiment_path_roots(experiment_dir):
                    js = root / "results" / str(job_id) / "job_script.slurm"
                    if js.is_file():
                        est_tp = infer_hybrid_vec_gflops_from_slurm_cli(
                            js.read_text(encoding="utf-8", errors="replace"), runtime
                        )
                        if est_tp > 0.0:
                            break
            if est_tp > 0.0:
                throughput = est_tp
                logger.warning(
                    "  ⚠️  GFLOPS estimated from --size/--iterations vs wall clock (diagnostic only when wall fallback is used)."
                )

        if runtime <= 0.0 and (
            "Exec format error" in slurm_content
            or "cannot execute binary file" in slurm_content
        ):
            logger.error(
                "  Executable did not run on compute nodes (Exec format error / wrong architecture). "
                "Fix the binary; wall-clock fallback also failed or is missing Start/End lines."
            )

        if num_profiles == 0 and cpu_only and cpu_utilization <= 0.0 and runtime > 0.0:
            cpu_utilization = 0.92
            logger.info(
                "  β (thread): no Nsight SQLite — using neutral CPU busy 0.92 for scoring "
                "(set --enable-likwid / run Phase-2 with nsys; or ensure results/<job>/profiling/profile.sqlite exists)."
            )

        # LULESH `-q` or missing tee: no FOM in logs — derive z/s from job script -s/-i.
        if throughput <= 0.0 and runtime > 0.0:
            js_path = experiment_dir / "results" / job_id / "job_script.slurm"
            sm, it = parse_lulesh_mesh_iter_from_job_script(js_path)
            derived_zps = lulesh_throughput_z_per_s_from_mesh(sm, it, runtime)
            if derived_zps > 0.0:
                throughput = derived_zps
                logger.info(
                    f"  LULESH throughput from mesh (mesh³×iter/runtime, job script): {throughput:.2f} z/s"
                )

        rpn = int(config['mpi_ranks_per_node'])
        row_nodes = config.get("num_nodes")
        if row_nodes is None:
            row_nodes = num_nodes
        else:
            row_nodes = int(row_nodes)
        row_nodes = max(1, row_nodes)
        config_metrics = {
            'config_name': config_name,
            'mpi_ranks': rpn,
            'num_nodes': row_nodes,
            'total_mpi_ranks': row_nodes * rpn,
            'omp_threads': config['omp_threads_per_rank'],
            'total_cores': config['mpi_ranks_per_node'] * config['omp_threads_per_rank'],
            'runtime_sec': runtime,
            'throughput_gflops': throughput,
            # Real metrics (from NsightMetricsExtractor) - use 0.0 for missing data (not None)
            'mpi_comm_time': mpi_comm_time
            if (mpi_comm_time > 0 or num_profiles > 0)
            else 0.0,
            'mpip_app_time': mpip_app_time
            if (mpip_app_time > 0 or mpip_max_mpi_wall_fraction > 0 or mpi_comm_time > 0 or num_profiles > 0)
            else 0.0,
            'mpip_max_mpi_wall_fraction': mpip_max_mpi_wall_fraction
            if (mpip_max_mpi_wall_fraction > 0 or mpi_comm_time > 0 or num_profiles > 0)
            else 0.0,
            'gpu_busy_time': gpu_busy_time if (gpu_busy_time > 0 or num_profiles > 0) else 0.0,
            'cpu_utilization': cpu_utilization if (cpu_utilization > 0 or num_profiles > 0) else 0.0,
            # For pure MPI, always store 0.0; for others, use extracted value or 0.0 if missing
            'openmp_work_efficiency': openmp_work_efficiency if (config['omp_threads_per_rank'] == 1 or openmp_work_efficiency > 0 or num_profiles > 0) else 0.0,
            'openmp_instrumented': openmp_instrumented,
            'gpu_utilization': gpu_utilization if (gpu_utilization > 0 or num_profiles > 0) else 0.0,
            'gpu_utilization_source': gpu_utilization_source if num_profiles > 0 else "no_gpu_metrics",
            'gpu_utilization_cupti': gpu_utilization_cupti if (gpu_utilization_cupti > 0 or num_profiles > 0) else 0.0,
            'nvidia_smi_gpu_util': nvidia_smi_gpu_util if (nvidia_smi_gpu_util > 0 or num_profiles > 0) else 0.0,
            'profiling_phase_aligned': profiling_phase_aligned,
            'numa_efficiency': numa_efficiency if (numa_efficiency > 0 or num_profiles > 0) else 0.0,
            'locality_source': locality_source,
            'thread_stall_time': thread_stall_time if (thread_stall_time > 0 or num_profiles > 0) else 0.0,
            'memory_bandwidth': memory_bandwidth
            if (memory_bandwidth > 0 or num_profiles > 0)
            else 0.0,
            'num_rank_profiles': num_profiles,
            'job_id': job_id,
            'profiling_runtime_used': profiling_runtime_used,
            'runtime_wall_fallback': wall_runtime_used,
            'profile_total_runtime': profile_total_runtime,  # Nsight trace duration for α comm_fraction
        }
        
        all_metrics.append(config_metrics)
    
    # Deduplicate configs: keep only the LATEST job (highest job_id) for each config_name
    # This handles the case where phase2_job_ids.json is missing and both Phase 1 and Phase 2 jobs exist
    if all_metrics:
        # Group by config_name
        config_groups = {}
        for m in all_metrics:
            name = m['config_name']
            if name not in config_groups:
                config_groups[name] = []
            config_groups[name].append(m)
        
        # Check if we have duplicates
        has_duplicates = any(len(v) > 1 for v in config_groups.values())
        if has_duplicates:
            logger.info("📋 Deduplicating configs (keeping latest job ID for each config)...")
            deduplicated = []
            for name, metrics_list in config_groups.items():
                if len(metrics_list) > 1:
                    # Sort by job_id descending (highest = latest = Phase 2)
                    metrics_list.sort(key=lambda x: int(x.get('job_id', 0)), reverse=True)
                    latest = metrics_list[0]
                    logger.info(f"   {name}: keeping job {latest['job_id']} (discarding {len(metrics_list)-1} older jobs)")
                    deduplicated.append(latest)
                else:
                    deduplicated.append(metrics_list[0])
            all_metrics = deduplicated
            logger.info(f"✅ After deduplication: {len(all_metrics)} unique configs")
    
    return all_metrics


def _log_binary_failure_hints_from_slurm(experiment_dir: Path) -> None:
    """Emit a single actionable hint when logs show the binary never executed on compute nodes."""
    for path in sorted(gather_slurm_logs_by_job_id(experiment_dir).values()):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:20000]
            except OSError:
                continue
            if "Exec format error" in text or "cannot execute binary file" in text.lower():
                logger.error(
                    "SLURM logs show the executable did not run on compute nodes (Exec format error / "
                    "wrong architecture). Rebuild on the target system. If Start/End wall lines are missing, "
                    "regenerate cannot infer runtime."
                )
                return


def generate_plots(experiment_dir: Path, all_metrics: List[Dict[str, Any]], 
                   total_cores: int, enable_locality: bool = False, cpu_only: bool = True) -> Path:
    """Generate visualization plots from collected metrics with 5 heuristic components."""
    if not all_metrics:
        logger.error("No metrics to plot")
        return None
    
    output_dir = experiment_dir / "comprehensive_results"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        # Try alternative location
        try:
            output_dir = Path.cwd() / "comprehensive_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using alternative output directory: {output_dir}")
        except Exception as e2:
            logger.error(f"Failed to create alternative output directory: {e2}")
            raise
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Ensure mpi_ranks and omp_threads are numeric (so pure-MPI / OpenMP masks work correctly)
    for col in ('mpi_ranks', 'omp_threads'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)

    # Global MPI rank count for α–ε masks: nodes × mpi_ranks_per_node (`mpi_ranks` column is per node).
    if 'num_nodes' not in df.columns:
        df['num_nodes'] = 1
    df['num_nodes'] = pd.to_numeric(df['num_nodes'], errors='coerce').fillna(1).astype(int).clip(lower=1)
    if 'total_mpi_ranks' not in df.columns:
        df['total_mpi_ranks'] = df['mpi_ranks'] * df['num_nodes']
    else:
        df['total_mpi_ranks'] = pd.to_numeric(df['total_mpi_ranks'], errors='coerce')
        bad = df['total_mpi_ranks'].isna() | (df['total_mpi_ranks'] <= 0)
        df.loc[bad, 'total_mpi_ranks'] = (
            df.loc[bad, 'mpi_ranks'].astype(float) * df.loc[bad, 'num_nodes'].astype(float)
        )
    df['total_mpi_ranks'] = (
        pd.to_numeric(df['total_mpi_ranks'], errors='coerce')
        .fillna(df['mpi_ranks'] * df['num_nodes'])
        .astype(int)
        .clip(lower=1)
    )
    
    # Calculate derived metrics with validation
    # Ensure non-negative values
    df['runtime_sec'] = df['runtime_sec'].clip(lower=0)
    
    
    # Calculate scores
    max_runtime = df['runtime_sec'].max()
    min_runtime = df['runtime_sec'].min()
    
    if max_runtime > min_runtime:
        df['runtime_score'] = 1 - (df['runtime_sec'] - min_runtime) / (max_runtime - min_runtime + 0.001)
    else:
        df['runtime_score'] = 1.0
    
    max_throughput = df['throughput_gflops'].max()
    df['throughput_score'] = df['throughput_gflops'] / (max_throughput + 0.001) if max_throughput > 0 else 0
    
    # Calculate communication score (alpha) with proper differentiation
    # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios (use global rank count)
    is_pure_mpi = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
    is_pure_openmp = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] > 1)
    is_hybrid = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] > 1)
    
    # For pure OpenMP: No MPI communication, so communication score should be high
    df['comm_score'] = 0.0  # Initialize
    df.loc[is_pure_openmp, 'comm_score'] = 1.0
    
    # For α (comm_fraction): prefer mpiP AppTime denominator when available (same source as mpi_comm_time)
    if 'mpip_app_time' not in df.columns:
        df['mpip_app_time'] = 0.0
    df['mpip_app_time'] = pd.to_numeric(df['mpip_app_time'], errors='coerce').fillna(0.0)
    if 'mpip_max_mpi_wall_fraction' not in df.columns:
        df['mpip_max_mpi_wall_fraction'] = 0.0
    df['mpip_max_mpi_wall_fraction'] = pd.to_numeric(
        df['mpip_max_mpi_wall_fraction'], errors='coerce'
    ).fillna(0.0).clip(0.0, 1.0)

    # Fallback denominator uses profile trace duration when mpiP AppTime is unavailable.
    if 'profile_total_runtime' not in df.columns:
        df['profile_total_runtime'] = 0.0
    df['profile_total_runtime'] = pd.to_numeric(df['profile_total_runtime'], errors='coerce').fillna(0.0)
    
    # For pure MPI: Use REAL mpi_comm_time from profiling (not OSRT event counts)
    if is_pure_mpi.any():
        pure_mpi_df = df[is_pure_mpi].copy()
        if len(pure_mpi_df) > 0:
            # Use real MPI communication time if available
            has_real_comm = pure_mpi_df['mpi_comm_time'].notna() & (pure_mpi_df['mpi_comm_time'] > 0)
            if has_real_comm.any():
                comm_time = pure_mpi_df.loc[has_real_comm, 'mpi_comm_time'].values
                app_time  = pure_mpi_df.loc[has_real_comm, 'mpip_app_time'].values
                wall_frac = pure_mpi_df.loc[has_real_comm, 'mpip_max_mpi_wall_fraction'].fillna(0.0).values
                # nsight_mpi_sum normalization: when Nsight is the only MPI-time source
                # (no mpiP wall fraction, no mpiP app time), the stored mpi_comm_time is the
                # SUM of all MPI call durations across ALL ranks — not a per-rank value.
                # Dividing by total_mpi_ranks gives the per-rank average, which is the correct
                # scale for comm_fraction = comm_time / runtime.  mpiP-sourced rows (wall_frac
                # or app_time present) are left unchanged.
                _nsight_only_mask = (wall_frac <= 0.001) & (app_time <= 0.001)
                if _nsight_only_mask.any():
                    _ranks = pure_mpi_df.loc[has_real_comm, 'total_mpi_ranks'].values.astype(float)
                    _ranks = np.maximum(_ranks, 1.0)
                    comm_time = np.where(_nsight_only_mask, comm_time / _ranks, comm_time)
                # Validate mpip_app_time: mpiP AppTime must always exceed MPITime by a
                # meaningful margin (app time includes computation + MPI). If the stored
                # value is ≤ mpi_comm_time it is likely storing MPI time again due to a
                # column-swap in the mpiP report — fall back to Nsight trace / wall clock.
                app_time_valid = (app_time > comm_time + 1e-4) & (app_time > 0.001)
                # Denominator priority: validated mpiP AppTime > profile trace > runtime
                denom = np.where(
                    app_time_valid,
                    app_time,
                    np.where(
                        pure_mpi_df.loc[has_real_comm, 'profile_total_runtime'].values > 0.001,
                        pure_mpi_df.loc[has_real_comm, 'profile_total_runtime'].values,
                        pure_mpi_df.loc[has_real_comm, 'runtime_sec'].values
                    )
                )
                denom = np.maximum(denom, 0.001)
                comm_time_capped = np.minimum(comm_time, denom)
                ratio_frac = (comm_time_capped / denom).clip(0.0, 1.0)
                # Use min(wall_frac, ratio_frac) when both are meaningful.
                # If the stored wall_frac contains app_time-in-seconds (field-mapping bug),
                # it will be >> ratio_frac and the min picks the correct ratio value.
                # If wall_frac is a legitimate per-rank fraction (e.g. LULESH 8x6: 0.327),
                # it is typically ≤ ratio_frac and the min preserves it as the better estimate.
                mpi_comm_fraction = np.where(
                    (wall_frac > 0.001) & (ratio_frac > 0.001),
                    np.minimum(np.minimum(wall_frac, 1.0), ratio_frac),
                    np.where(wall_frac > 0.001, np.minimum(wall_frac, 1.0), ratio_frac)
                )
                pure_mpi_df.loc[has_real_comm, 'mpi_comm_fraction'] = mpi_comm_fraction
                # Communication score = 1 - comm_fraction (lower comm = better)
                pure_mpi_df.loc[has_real_comm, 'comm_score'] = (
                    1.0 - pure_mpi_df.loc[has_real_comm, 'mpi_comm_fraction']
                )
            # No fallback: configs without real comm data get 0.0 (α from profiling only)
            if (~has_real_comm).any():
                pure_mpi_df.loc[~has_real_comm, 'comm_score'] = 0.0
            
            # Apply small penalty for very high rank counts (communication complexity)
            pure_mpi_df.loc[pure_mpi_df['total_mpi_ranks'] > 16, 'comm_score'] *= 0.95
            
            df.loc[is_pure_mpi, 'comm_score'] = pure_mpi_df['comm_score']
    
    # For hybrid: Use REAL mpi_comm_time + overlap bonus
    if is_hybrid.any():
        hybrid_df = df[is_hybrid].copy()
        if len(hybrid_df) > 0:
            # Use real MPI communication time if available
            has_real_comm = hybrid_df['mpi_comm_time'].notna() & (hybrid_df['mpi_comm_time'] > 0)
            # Initialize base_comm for all hybrid configs
            base_comm = pd.Series(index=hybrid_df.index, dtype=float)
            
            if has_real_comm.any():
                comm_time = hybrid_df.loc[has_real_comm, 'mpi_comm_time'].values
                app_time  = hybrid_df.loc[has_real_comm, 'mpip_app_time'].values
                wall_frac_h = hybrid_df.loc[has_real_comm, 'mpip_max_mpi_wall_fraction'].fillna(0.0).values
                # nsight_mpi_sum normalization (same logic as pure-MPI path above).
                _nsight_only_mask_h = (wall_frac_h <= 0.001) & (app_time <= 0.001)
                if _nsight_only_mask_h.any():
                    _ranks_h = hybrid_df.loc[has_real_comm, 'total_mpi_ranks'].values.astype(float)
                    _ranks_h = np.maximum(_ranks_h, 1.0)
                    comm_time = np.where(_nsight_only_mask_h, comm_time / _ranks_h, comm_time)
                # Same AppTime validation as pure-MPI path above.
                app_time_valid = (app_time > comm_time + 1e-4) & (app_time > 0.001)
                # Denominator priority: validated mpiP AppTime > profile trace > runtime
                denom = np.where(
                    app_time_valid,
                    app_time,
                    np.where(
                        hybrid_df.loc[has_real_comm, 'profile_total_runtime'].fillna(0).values > 0.001,
                        hybrid_df.loc[has_real_comm, 'profile_total_runtime'].values,
                        hybrid_df.loc[has_real_comm, 'runtime_sec'].values
                    )
                )
                denom = np.maximum(denom, 0.001)
                comm_time_capped = np.minimum(comm_time, denom)
                ratio_frac = (comm_time_capped / denom).clip(0.0, 1.0)
                wall_frac = wall_frac_h
                mpi_comm_fraction = np.where(
                    (wall_frac > 0.001) & (ratio_frac > 0.001),
                    np.minimum(np.minimum(wall_frac, 1.0), ratio_frac),
                    np.where(wall_frac > 0.001, np.minimum(wall_frac, 1.0), ratio_frac)
                )
                hybrid_df.loc[has_real_comm, 'mpi_comm_fraction'] = mpi_comm_fraction
                base_comm.loc[has_real_comm] = 1.0 - hybrid_df.loc[has_real_comm, 'mpi_comm_fraction']
            
            # No fallback: configs without real comm data get 0.0 (α from profiling only)
            if (~has_real_comm).any():
                base_comm.loc[~has_real_comm] = 0.0
            
            # Fill any NaN values (missing = no real data)
            base_comm = base_comm.fillna(0.0)
            
            # MPI overhead penalty: scales with rank count, attenuated when GPU util is high (MPI less likely
            # the sole bottleneck on hybrid GPU runs — measured signal, not a fixed table).
            mpi_penalty_base = np.minimum(0.20, (hybrid_df['total_mpi_ranks'].values - 1) * 0.02)
            gpu_u = hybrid_df['gpu_utilization'].fillna(0.0).clip(0.0, 1.0).values
            if cpu_only:
                mpi_penalty = mpi_penalty_base
            else:
                mpi_penalty = mpi_penalty_base * (1.0 - 0.45 * gpu_u)
            mpi_penalty = np.clip(mpi_penalty, 0.0, 0.20)

            # Overlap bonus: blend thread-count prior with CPU×GPU concurrency (data-driven on GPU workloads).
            thr = hybrid_df['omp_threads'].values
            thread_prior = np.where(
                thr > 1,
                np.minimum(0.08, np.sqrt(thr / 2.0) * 0.04),
                0.0,
            )
            cpu_u = hybrid_df['cpu_utilization'].fillna(0.0).clip(0.0, 1.0).values
            if cpu_only:
                # CPU-only: no GPU to overlap with MPI transfers; assume zero measurable overlap.
                # Leaving a non-zero thread_prior bonus here inflates α above 1.0 for any hybrid
                # config whose MPI overhead is small (e.g. 2x24 with ~2% MPI → clips to 1.000).
                overlap_bonus = np.zeros(len(hybrid_df))
            else:
                data_overlap = np.minimum(0.10, 0.22 * gpu_u * cpu_u)
                overlap_bonus = np.minimum(0.10, 0.40 * thread_prior + 0.60 * data_overlap)
            
            hybrid_df['comm_score'] = base_comm * (1.0 - mpi_penalty + overlap_bonus)
            df.loc[is_hybrid, 'comm_score'] = hybrid_df['comm_score']
    
    # Fallback for single rank, single thread (shouldn't happen in practice)
    df.loc[(df['total_mpi_ranks'] == 1) & (df['omp_threads'] == 1), 'comm_score'] = 1.0
    
    # Ensure comm_score is in valid range
    df['comm_score'] = df['comm_score'].clip(0.0, 1.0)
    
    # Transparency: how α was derived (mpiP wall-fraction vs ratio vs Nsight MPI table)
    df['alpha_source'] = 'missing'
    df.loc[is_pure_openmp, 'alpha_source'] = 'not_in_composite'
    mpi_axes = is_pure_mpi | is_hybrid
    has_comm = mpi_axes & (pd.to_numeric(df['mpi_comm_time'], errors='coerce').fillna(0.0) > 0)
    # mpip_wall_fraction: mpiP provided per-rank max(MPI/wall) directly — most accurate.
    df.loc[has_comm & (df['mpip_max_mpi_wall_fraction'] > 0.001), 'alpha_source'] = 'mpip_wall_fraction'
    # mpip_time_ratio: mpiP provided comm time but not per-rank wall fraction; use time/apptime ratio.
    has_mpip_time = has_comm & (df['mpip_max_mpi_wall_fraction'] <= 0.001) & (
        pd.to_numeric(df.get('mpip_app_time', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0) > 0.001
    )
    df.loc[has_mpip_time, 'alpha_source'] = 'mpip_time_ratio'
    # nsight_mpi_avg: no mpiP data; comm time from Nsight MPI event table, normalized by total_mpi_ranks
    # to convert the rank-summed value into a per-rank average for a fair α estimate.
    has_nsight_only = has_comm & (df['mpip_max_mpi_wall_fraction'] <= 0.001) & ~has_mpip_time
    df.loc[has_nsight_only, 'alpha_source'] = 'nsight_mpi_avg'

    # Efficiency score: same scale as runtime_score across configs
    df['efficiency_score'] = df['runtime_score'].copy()
    
    
    # Use detected total_cores instead of hardcoded 48
    df['scalability_score'] = (df['omp_threads'] / float(total_cores)) * df['runtime_score']
    
    # Calculate 5 heuristic components (α, β, γ, δ, ε)
    # α = Communication efficiency (inverse of comm overhead)
    df['alpha_comm'] = df['comm_score']
    
    # β = Thread efficiency: prefer conservative agreement between CPU util and stall-derived busy time.
    base_thread_score = df['cpu_utilization'].copy().fillna(0.0).clip(0.0, 1.0)
    df['beta_source'] = 'cpu_only'
    if 'thread_stall_time' in df.columns:
        stall = pd.to_numeric(df['thread_stall_time'], errors='coerce').fillna(0.0).clip(lower=0.0)
        # Denominator priority for stall fraction: mpiP AppTime > profile trace runtime > application runtime.
        beta_denom = np.where(
            df['mpip_app_time'].values > 0.001,
            df['mpip_app_time'].values,
            np.where(
                df['profile_total_runtime'].values > 0.001,
                df['profile_total_runtime'].values,
                np.maximum(df['runtime_sec'].values, 0.001)
            )
        )
        beta_denom = np.maximum(beta_denom, 0.001)
        stall_busy = (1.0 - (np.minimum(stall.values, beta_denom) / beta_denom)).clip(0.0, 1.0)
        # Conservative fusion: take the lower of CPU-util and stall-derived busy fraction.
        fused_thread = np.minimum(base_thread_score.values, stall_busy)
        has_stall_signal = (stall.values > 0)
        base_thread_score = pd.Series(np.where(has_stall_signal, fused_thread, base_thread_score.values), index=df.index)
        df.loc[has_stall_signal, 'beta_source'] = 'cpu_and_stall'
    
    # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
    is_pure_mpi = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
    is_pure_openmp = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] > 1)
    is_hybrid = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] > 1)
    
    # Thread balance factor - only applies when OpenMP parallelism exists
    df['thread_balance_factor'] = 1.0  # Initialize
    
    # Pure MPI: No OpenMP threads to balance, so no penalty
    # (already 1.0, so no change needed)
    
    # Pure OpenMP or Hybrid: Apply thread balance factor
    openmp_configs = is_pure_openmp | is_hybrid
    df.loc[openmp_configs & (df['omp_threads'] >= 4), 'thread_balance_factor'] = 1.0
    df.loc[openmp_configs & (df['omp_threads'] == 1), 'thread_balance_factor'] = 0.7
    df.loc[openmp_configs & (df['omp_threads'] == 2), 'thread_balance_factor'] = 0.8
    df.loc[openmp_configs & (df['omp_threads'] == 3), 'thread_balance_factor'] = 0.9
    
    # Upper bound penalty for excessive threads (>32)
    excess_mask = openmp_configs & (df['omp_threads'] > 32)
    if excess_mask.any():
        excess_penalty = np.minimum(0.15, (df.loc[excess_mask, 'omp_threads'] - 32) * 0.005)
        df.loc[excess_mask, 'thread_balance_factor'] = df.loc[excess_mask, 'thread_balance_factor'] * (1.0 - excess_penalty)
    
    # MPI sync overhead factor - only applies when MPI parallelism exists
    df['sync_overhead_factor'] = 1.0  # Initialize
    
    # Pure OpenMP: No MPI synchronization, so no penalty
    # (already 1.0, so no change needed)
    
    # Pure MPI or Hybrid: Apply sync overhead with logarithmic scaling
    mpi_configs = is_pure_mpi | is_hybrid
    df.loc[mpi_configs & (df['total_mpi_ranks'] == 1), 'sync_overhead_factor'] = 1.0
    df.loc[mpi_configs & (df['total_mpi_ranks'] > 1), 'sync_overhead_factor'] = np.maximum(
        0.7, 1.0 - np.log2(df.loc[mpi_configs & (df['total_mpi_ranks'] > 1), 'total_mpi_ranks']) * 0.01
    )
    
    # Use weighted combination instead of multiplicative (less harsh)
    df['beta_thread'] = base_thread_score * (0.7 * df['thread_balance_factor'] + 0.3 * df['sync_overhead_factor'])
    df['beta_thread'] = df['beta_thread'].clip(0.0, 1.0)
    
    # γ = Locality - Use REAL numa_efficiency from profiling if available
    # γ = Locality - Priority: Memory Bandwidth > NUMA Efficiency > Fallback
    logger.debug(f"Entering Locality Calculation. enable_locality={enable_locality}")
    
    if enable_locality:
        # No fallback: initialize to 0.0; only real NUMA/bandwidth data set γ
        base_locality_score = pd.Series([0.0] * len(df), index=df.index)
        
        # Check for Bandwidth Data (Primary Metric)
        # -------------------------------------------------------------------------
        if 'memory_bandwidth' in df.columns:
            # Ensure numeric and handle NaNs
            df['memory_bandwidth'] = pd.to_numeric(df['memory_bandwidth'], errors='coerce').fillna(0.0)
            max_bw = df['memory_bandwidth'].max()
            
            # Use threshold to ignore noise (e.g. < 0.05 GB/s)
            if max_bw > 0.05:
                logger.debug(f"Found valid Memory Bandwidth data. Max={max_bw:.4f} GB/s")
                
                # Per-config floor (same scale as max_bw check): Nsight/LIKWID sometimes emits
                # tiny positive CPU BW (e.g. <0.1 GB/s) that is not comparable to DRAM-class
                # values on other configs — do not use it for γ or those rows never reach NUMA.
                bw_floor = 0.05
                valid_bw_mask = df['memory_bandwidth'] > bw_floor
                if valid_bw_mask.any():
                    base_locality_score.loc[valid_bw_mask] = (
                        df.loc[valid_bw_mask, 'memory_bandwidth'] / max_bw
                    ).clip(0.0, 1.0)
                    logger.debug(
                        "Using Bandwidth-based Locality Score where memory_bandwidth > %.1f GB/s",
                        bw_floor,
                    )
            else:
                logger.warning(f"Memory Bandwidth max is {max_bw} (too low). checking NUMA...")
        
        # Use NUMA Efficiency where Bandwidth missing — or override the BW estimate when
        # high-quality numastat data is available (locality_source == "numa_meminfo").
        # The bandwidth ratio is heuristic; numastat is a direct page-allocation measurement.
        # -------------------------------------------------------------------------
        missing_bw_mask = (base_locality_score == 0.0)

        # Also override BW-based score for configs whose locality_source is "numa_meminfo"
        # (numastat data is higher-quality than bandwidth/max_bw ratio).
        if 'locality_source' in df.columns:
            has_numastat = df['locality_source'] == 'numa_meminfo'
            if has_numastat.any():
                missing_bw_mask = missing_bw_mask | has_numastat
                logger.debug(
                    f"numastat data available for {has_numastat.sum()} config(s) — "
                    "overriding BW estimate with real NUMA page-allocation fraction."
                )

        if missing_bw_mask.any():
            # Force numeric conversion for NUMA efficiency
            if 'numa_efficiency' in df.columns:
                # Check raw values first
                logger.debug(f"Raw numa_efficiency head: {df['numa_efficiency'].head().tolist()}")
                df['numa_efficiency'] = pd.to_numeric(df['numa_efficiency'], errors='coerce')
                logger.debug(f"Coerced numa_efficiency head: {df['numa_efficiency'].head().tolist()}")
            else:
                logger.warning("numa_efficiency column MISSING from DataFrame!")
                df['numa_efficiency'] = np.nan
                 
            # Note: 0.0 means missing, so we look for > 0
            has_real_numa = df['numa_efficiency'].notna() & (df['numa_efficiency'] > 0)
            apply_numa_mask = missing_bw_mask & has_real_numa
            logger.debug(f"Applying real NUMA data to {apply_numa_mask.sum()} configs")

            if apply_numa_mask.any():
                # Real metric from LIKWID or profiling
                base_locality_score.loc[apply_numa_mask] = df.loc[apply_numa_mask, 'numa_efficiency']
                logger.debug("Using REAL NUMA data for γ (locality).")
            else:
                logger.debug("No real NUMA data; γ remains 0.0 (no fallback).")
            
        logger.debug(f"Base locality score head: {base_locality_score.head().tolist()}")
        

        
        # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
        is_pure_mpi = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
        is_pure_openmp = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] > 1)
        is_hybrid = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] > 1)
        
        # Cache locality factor (OpenMP-related) - only applies when OpenMP parallelism exists
        df['cache_locality_factor'] = 1.0  # Initialize
        
        # Pure MPI: No OpenMP threads, so cache locality not applicable
        # (already 1.0, so no change needed)
        
        # Pure OpenMP or Hybrid: Apply cache locality factor
        openmp_configs = is_pure_openmp | is_hybrid
        # Gradual scaling for 1-8 threads
        df.loc[openmp_configs & (df['omp_threads'] <= 8), 'cache_locality_factor'] = (
            0.7 + (df.loc[openmp_configs & (df['omp_threads'] <= 8), 'omp_threads'] - 1) * (0.3 / 7.0)
        )
        # Optimal range for 9-32 threads
        df.loc[openmp_configs & (df['omp_threads'] > 8) & (df['omp_threads'] <= 32), 'cache_locality_factor'] = 1.0
        # Upper bound penalty for >32 threads
        excess_mask = openmp_configs & (df['omp_threads'] > 32)
        if excess_mask.any():
            excess_penalty = np.minimum(0.20, (df.loc[excess_mask, 'omp_threads'] - 32) * 0.005)
            df.loc[excess_mask, 'cache_locality_factor'] = 1.0 - excess_penalty
        
        # NUMA distribution factor (MPI-related) - only applies when MPI parallelism exists
        df['memory_distribution_factor'] = 1.0  # Initialize
        
        # Pure OpenMP: No MPI distribution, so NUMA factor doesn't apply
        # (already 1.0, so no change needed)
        
        # Pure MPI or Hybrid: Apply NUMA distribution factor with logarithmic scaling
        mpi_configs = is_pure_mpi | is_hybrid
        df.loc[mpi_configs & (df['total_mpi_ranks'] == 1), 'memory_distribution_factor'] = 1.0
        df.loc[mpi_configs & (df['total_mpi_ranks'] > 1), 'memory_distribution_factor'] = np.maximum(
            0.7, 1.0 - np.log2(df.loc[mpi_configs & (df['total_mpi_ranks'] > 1), 'total_mpi_ranks']) * 0.03
        )
        
        # Apply only relevant factors based on configuration type
        df['gamma_locality'] = 0.0  # Initialize
        
        # Ensure base_locality_score is a Series for proper alignment
        if isinstance(base_locality_score, pd.Series):
            base_locality_series = base_locality_score
        else:
            base_locality_series = pd.Series(base_locality_score, index=df.index)
        
        # Pure OpenMP: Only cache locality matters
        df.loc[is_pure_openmp, 'gamma_locality'] = (
            base_locality_series.loc[is_pure_openmp] * df.loc[is_pure_openmp, 'cache_locality_factor']
        )
        
        # Pure MPI: Only NUMA distribution matters
        df.loc[is_pure_mpi, 'gamma_locality'] = (
            base_locality_series.loc[is_pure_mpi] * df.loc[is_pure_mpi, 'memory_distribution_factor']
        )
        
        # Hybrid: Both matter, use weighted combination
        df.loc[is_hybrid, 'gamma_locality'] = base_locality_series.loc[is_hybrid] * (
            0.6 * df.loc[is_hybrid, 'cache_locality_factor'] + 
            0.4 * df.loc[is_hybrid, 'memory_distribution_factor']
        )
        
        # Fallback for single rank, single thread
        single_config = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] == 1)
        df.loc[single_config, 'gamma_locality'] = base_locality_series.loc[single_config]
        
        df['gamma_locality'] = df['gamma_locality'].clip(0.0, 1.0)
    else:
        df['gamma_locality'] = 0.0  # Disabled for this application
    
    # δ = GPU utilization - Use REAL gpu_utilization from profiling if available
    if cpu_only:
        df['delta_gpu'] = 0.0
        df['delta_gpu_source'] = 'cpu_only_app'
    else:
        df['delta_gpu_source'] = 'no_gpu_metrics'
        if 'nvidia_smi_gpu_util' not in df.columns:
            df['nvidia_smi_gpu_util'] = 0.0
        df['nvidia_smi_gpu_util'] = pd.to_numeric(df['nvidia_smi_gpu_util'], errors='coerce').fillna(0.0)
        if 'profiling_phase_aligned' not in df.columns:
            df['profiling_phase_aligned'] = True

        # Use real GPU utilization from profiling if available
        has_real_gpu = df['gpu_utilization'].notna() & (df['gpu_utilization'] > 0)
        if has_real_gpu.any():
            df['delta_gpu'] = df['gpu_utilization'].fillna(0.0).clip(0.0, 1.0)
        else:
            df['delta_gpu'] = 0.0
        
        if 'gpu_utilization' in df.columns and df['gpu_utilization'].notna().any():
            base_gpu = df['gpu_utilization'].fillna(0.0)

            # One profile.sqlite per job = full-job Nsight (all ranks); num_rank_profiles counts sqlite files, not ranks.
            if 'num_rank_profiles' in df.columns:
                sqlite_n = pd.to_numeric(df['num_rank_profiles'], errors='coerce').fillna(0.0)
            else:
                sqlite_n = pd.Series([0.0] * len(df), index=df.index)
            df['delta_profile_coverage'] = (sqlite_n >= 1.0).astype(float)

            # Small multi-rank bump only when coverage is high (policy scaled by measurement reach).
            multi_gpu_mask = (df['total_mpi_ranks'] > 1) & (base_gpu > 0.7)
            if multi_gpu_mask.any():
                bump = np.minimum(
                    0.02,
                    (df.loc[multi_gpu_mask, 'total_mpi_ranks'].values - 1) * 0.001,
                ) * df.loc[multi_gpu_mask, 'delta_profile_coverage'].values
                df.loc[multi_gpu_mask, 'delta_gpu'] = base_gpu.loc[multi_gpu_mask] + bump

            # CPU–GPU coordination: only strong GPU with very low CPU host hints at launch/serial issues.
            hybrid_mask = (df['omp_threads'] > 1) & (base_gpu > 0.85) & (df['cpu_utilization'] < 0.15)
            if hybrid_mask.any():
                df.loc[hybrid_mask, 'delta_gpu'] = base_gpu.loc[hybrid_mask] * 0.96

        if 'gpu_utilization_source' in df.columns:
            refined = df['gpu_utilization_source'].fillna('').astype(str).str.len() > 0
            df.loc[refined, 'delta_gpu_source'] = df.loc[refined, 'gpu_utilization_source']
        else:
            smi_ok = df['nvidia_smi_gpu_util'] > 0.02
            mixed_mpi = pd.to_numeric(df['total_mpi_ranks'], errors='coerce').fillna(1.0) > 1
            misaligned = ~df['profiling_phase_aligned'].fillna(True)
            df['delta_gpu_source'] = np.where(smi_ok, 'cupt_nvsmi_geomean', 'cupt_time_ratio')
            df['delta_gpu_source'] = np.where(
                mixed_mpi,
                np.where(
                    misaligned & smi_ok,
                    'rank0_mpi_cupt_nvsmi_no_wall_align',
                    np.where(smi_ok, 'rank0_mpi_cupt_nvsmi_geomean', 'rank0_mpi_cupt_only'),
                ),
                df['delta_gpu_source'],
            )
    
    # ε = OpenMP efficiency with configuration-aware adjustments
    # Fix 1 & 4: Use consistent calculation with heuristic_scoring.py
    
    # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
    is_pure_mpi = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
    is_pure_openmp = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] > 1)
    is_hybrid = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] > 1)
    
    # Fix 3: Pure MPI - epsilon shouldn't apply
    # Set to 0.0 (not 1.0) since weight is 0.0 anyway - this is clearer
    df['epsilon_openmp'] = 0.0  # Initialize
    df.loc[is_pure_mpi, 'epsilon_openmp'] = 0.0  # Pure MPI: no OpenMP, so epsilon = 0.0 (weight is also 0.0)
    
    # IMPORTANT: Ensure pure MPI configs stay at 0.0 (don't let later code override)
    # Mark pure MPI configs so they're not processed in OpenMP calculation
    pure_mpi_mask = is_pure_mpi
    
    # For OpenMP configs, calculate efficiency
    openmp_configs = is_pure_openmp | is_hybrid
    
    if openmp_configs.any():
        # ε from openmp_work_efficiency when present; otherwise scale from relative runtime across configs
        base_openmp = pd.Series(index=df.index, dtype=float)
        base_openmp.loc[:] = 0.0
        
        valid_openmp_mask = pd.Series(False, index=df.index)
        if 'openmp_work_efficiency' in df.columns:
            valid_openmp_mask = openmp_configs & df['openmp_work_efficiency'].notna() & (df['openmp_work_efficiency'] > 0.01)
        if valid_openmp_mask.any():
            base_openmp.loc[valid_openmp_mask] = df.loc[valid_openmp_mask, 'openmp_work_efficiency']
            logger.info(
                f"  OpenMP work efficiency from trace: {df.loc[valid_openmp_mask, 'config_name'].tolist()}"
            )
        missing_openmp_mask = openmp_configs & ~valid_openmp_mask
        if missing_openmp_mask.any():
            # Use runtime_score as a stand-in, but apply a floor (0.5) so very slow hybrid configs
            # don't get near-zero epsilon — slowness reflects compute/comm scaling, not OMP utilization.
            base_openmp.loc[missing_openmp_mask] = np.maximum(
                0.5, df.loc[missing_openmp_mask, 'runtime_score']
            )
            logger.debug(
                "  ε from relative runtime (no trace efficiency for): "
                f"{df.loc[missing_openmp_mask, 'config_name'].tolist()}"
            )
        
        short_eps_mask = pd.Series(False, index=df.index)
        if 'profile_total_runtime' in df.columns:
            prof_rt = pd.to_numeric(df['profile_total_runtime'], errors='coerce').fillna(0.0)
            short_eps_mask = openmp_configs & (prof_rt > 0) & (prof_rt < 0.10)
        elif 'runtime_sec' in df.columns:
            rt_col = pd.to_numeric(df['runtime_sec'], errors='coerce').fillna(0.0)
            short_eps_mask = openmp_configs & (rt_col > 0) & (rt_col < 0.10)

        # Fix 2 & 8: Configuration-based adjustments for thread count
        df['thread_factor'] = 1.0  # Initialize
        
        # 1-8 threads: Gradual scaling
        df.loc[openmp_configs & (df['omp_threads'] <= 8), 'thread_factor'] = (
            0.7 + (df.loc[openmp_configs & (df['omp_threads'] <= 8), 'omp_threads'] - 1) * (0.3 / 7.0)
        )
        # 9-32 threads: Optimal range
        df.loc[openmp_configs & (df['omp_threads'] > 8) & (df['omp_threads'] <= 32), 'thread_factor'] = 1.0
        # >32 threads: Upper bound penalty
        excess_mask = openmp_configs & (df['omp_threads'] > 32)
        if excess_mask.any():
            excess_penalty = np.minimum(0.20, (df.loc[excess_mask, 'omp_threads'] - 32) * 0.005)
            df.loc[excess_mask, 'thread_factor'] = 1.0 - excess_penalty
        
        # Fix 2: Consider MPI ranks for work distribution
        df['rank_factor'] = 1.0  # Initialize
        
        # Pure OpenMP: No MPI coordination overhead
        df.loc[is_pure_openmp, 'rank_factor'] = 1.0
        
        # Hybrid: MPI coordination overhead
        hybrid_mask = is_hybrid
        if hybrid_mask.any():
            # Logarithmic scaling for high rank counts (global MPI processes)
            df.loc[hybrid_mask & (df['total_mpi_ranks'] <= 8), 'rank_factor'] = (
                1.0 - (df.loc[hybrid_mask & (df['total_mpi_ranks'] <= 8), 'total_mpi_ranks'] - 1) * 0.01
            )
            df.loc[hybrid_mask & (df['total_mpi_ranks'] > 8), 'rank_factor'] = np.maximum(
                0.85, 1.0 - np.log2(df.loc[hybrid_mask & (df['total_mpi_ranks'] > 8), 'total_mpi_ranks']) * 0.01
            )
        
        # Fix 5: Use weighted combination (consistent with heuristic_scoring.py)
        # Calculate epsilon for each OpenMP config directly (avoid index alignment issues)
        for idx in df[openmp_configs].index:
            base_val = base_openmp.loc[idx]
            if (
                'openmp_instrumented' in df.columns
                and bool(df.loc[idx, 'openmp_instrumented'])
            ):
                df.loc[idx, 'epsilon_openmp'] = float(np.clip(base_val, 0.0, 1.0))
                logger.info(
                    f"  {df.loc[idx, 'config_name']}: epsilon={df.loc[idx, 'epsilon_openmp']:.3f} "
                    f"(OpenMP instrumented trace)"
                )
                continue
            if short_eps_mask.loc[idx]:
                df.loc[idx, 'epsilon_openmp'] = float(np.clip(base_val, 0.0, 1.0))
                logger.info(
                    f"  {df.loc[idx, 'config_name']}: epsilon={df.loc[idx, 'epsilon_openmp']:.3f} "
                    f"(trace-only; profile < 0.10s)"
                )
                continue
            thread_fac = df.loc[idx, 'thread_factor']
            rank_fac = df.loc[idx, 'rank_factor']
            mpi_frac_row = 0.0
            if 'mpip_max_mpi_wall_fraction' in df.columns:
                mpi_frac_row = float(
                    pd.to_numeric(df.loc[idx, 'mpip_max_mpi_wall_fraction'], errors='coerce') or 0.0
                )
            if mpi_frac_row <= 0.001 and 'mpi_comm_time' in df.columns and 'profile_total_runtime' in df.columns:
                mct = float(pd.to_numeric(df.loc[idx, 'mpi_comm_time'], errors='coerce') or 0.0)
                prt = float(pd.to_numeric(df.loc[idx, 'profile_total_runtime'], errors='coerce') or 0.0)
                if prt > 0 and mct > 0:
                    mpi_frac_row = min(1.0, mct / prt)
            adj = base_val
            if mpi_frac_row > 0.50 and is_hybrid.loc[idx]:
                adj = base_val * max(0.05, 1.0 - 0.75 * mpi_frac_row)
            df.loc[idx, 'epsilon_openmp'] = adj * (0.7 * thread_fac + 0.3 * rank_fac)
            
            config_name = df.loc[idx, 'config_name']
            logger.info(f"  {config_name}: epsilon={df.loc[idx, 'epsilon_openmp']:.3f} "
                      f"(base={base_val:.3f}, thread={thread_fac:.3f}, rank={rank_fac:.3f})")
        
        # For profiling plot and CSV: value used for ε
        df['openmp_work_efficiency_display'] = df['openmp_work_efficiency'].fillna(0.0)
        df.loc[openmp_configs, 'openmp_work_efficiency_display'] = base_openmp.loc[openmp_configs]
    
    # Fallback for single rank, single thread (NOT pure MPI - this is 1x1, not Nx1)
    # Only apply if NOT pure MPI (pure MPI already set to 0.0 above)
    single_config_mask = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] == 1) & ~pure_mpi_mask
    df.loc[single_config_mask, 'epsilon_openmp'] = 1.0
    
    # CRITICAL: Re-assert pure MPI configs have epsilon=0.0 (prevent any override)
    df.loc[pure_mpi_mask, 'epsilon_openmp'] = 0.0
    
    # Ensure epsilon is in valid range
    df['epsilon_openmp'] = df['epsilon_openmp'].clip(0.0, 1.0)
    
    # Final safeguard: pure MPI (N ranks x 1 thread) must always have epsilon_openmp = 0.0
    pure_mpi_final = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
    if pure_mpi_final.any():
        df.loc[pure_mpi_final, 'epsilon_openmp'] = 0.0
        logger.debug(f"Pure MPI configs (epsilon=0): {df.loc[pure_mpi_final, 'config_name'].tolist()}")
    
    # Ensure OpenMP display column exists for profiling plot (when no OpenMP configs, use raw column)
    if 'openmp_work_efficiency_display' not in df.columns:
        df['openmp_work_efficiency_display'] = df['openmp_work_efficiency'].fillna(0.0)
    
    
    # Calculate composite heuristic score with static baseline weights and profile layout redistribution
    # If locality is disabled, redistribute its weight
    if enable_locality:
        locality_weight = 0.20
        other_weight = 0.80
    else:
        locality_weight = 0.0
        other_weight = 1.0
    
    # If GPU is disabled, redistribute its weight
    if cpu_only:
        gpu_weight = 0.0
        other_weight_no_gpu = other_weight
    else:
        gpu_weight = 0.10
        other_weight_no_gpu = other_weight - gpu_weight
    
    # Normalize remaining weights (base weights before adaptive redistribution)
    remaining_weights = other_weight_no_gpu
    base_alpha_weight = 0.30 * (remaining_weights / 0.70)  # Scale from 0.30
    base_beta_weight = 0.30 * (remaining_weights / 0.70)   # Scale from 0.30
    base_epsilon_weight = 0.10 * (remaining_weights / 0.70) # Scale from 0.10
    
    # Adaptive weight redistribution for pure configurations
    # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
    is_pure_mpi = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
    is_pure_openmp = (df['total_mpi_ranks'] == 1) & (df['omp_threads'] > 1)
    is_hybrid = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] > 1)
    
    # Initialize adaptive weights
    df['effective_alpha'] = base_alpha_weight
    df['effective_epsilon'] = base_epsilon_weight
    
    # Pure OpenMP: Redistribute alpha weight to epsilon (MPI is irrelevant)
    df.loc[is_pure_openmp, 'effective_alpha'] = 0.0
    df.loc[is_pure_openmp, 'effective_epsilon'] = base_epsilon_weight + base_alpha_weight
    
    # Pure MPI: Redistribute epsilon weight to alpha (OpenMP is irrelevant)
    df.loc[is_pure_mpi, 'effective_alpha'] = base_alpha_weight + base_epsilon_weight
    df.loc[is_pure_mpi, 'effective_epsilon'] = 0.0
    
    # Hybrid: Use base weights (no redistribution)
    # (already set above, no change needed)
    
    # Normalize weights to ensure they sum to 1.0 for each configuration
    # Calculate total weight per row
    df['total_weight'] = df['effective_alpha'] + base_beta_weight + df['effective_epsilon'] + locality_weight + gpu_weight
    
    # Validate total_weight is not zero (should never happen, but protect against division by zero)
    if (df['total_weight'] <= 0).any():
        logger.error("ERROR: Total weight is zero or negative! This should never happen.")
        logger.error(f"  Weights: alpha={df['effective_alpha'].min():.4f}, beta={base_beta_weight:.4f}, "
                    f"epsilon={df['effective_epsilon'].min():.4f}, locality={locality_weight:.4f}, gpu={gpu_weight:.4f}")
        # Set minimum weight to prevent division by zero
        df['total_weight'] = df['total_weight'].clip(lower=0.001)
    
    # Normalize each weight component to ensure sum = 1.0
    df['effective_alpha'] = df['effective_alpha'] / df['total_weight']
    df['effective_epsilon'] = df['effective_epsilon'] / df['total_weight']
    effective_beta = base_beta_weight / df['total_weight']
    effective_locality = locality_weight / df['total_weight']
    effective_gpu = gpu_weight / df['total_weight']
    
    # Validate weights sum to 1.0 (with small tolerance for floating point)
    weight_sum = df['effective_alpha'] + effective_beta + df['effective_epsilon'] + effective_locality + effective_gpu
    if not np.allclose(weight_sum, 1.0, atol=0.001):
        logger.warning(f"⚠️  Weight sum is not exactly 1.0 (range: {weight_sum.min():.6f} - {weight_sum.max():.6f})")
        # Renormalize to fix any floating point errors
        df['effective_alpha'] = df['effective_alpha'] / weight_sum
        df['effective_epsilon'] = df['effective_epsilon'] / weight_sum
        effective_beta = effective_beta / weight_sum
        effective_locality = effective_locality / weight_sum
        effective_gpu = effective_gpu / weight_sum
    
    # Calculate composite heuristic score with adaptive weights
    df['heuristic_score'] = (
        df['effective_alpha'] * df['alpha_comm'] +
        effective_beta * df['beta_thread'] +
        effective_locality * df['gamma_locality'] +
        effective_gpu * df['delta_gpu'] +
        df['effective_epsilon'] * df['epsilon_openmp']
    )
    # Same value on a 0–10 scale (CSV/logs); components remain 0–1.
    df['heuristic_score_10'] = (df['heuristic_score'] * 10.0).clip(0.0, 10.0)
    
    # Clean up temporary columns
    df = df.drop(columns=['total_weight'], errors='ignore')
    
    # Separate valid and invalid configurations
    # Store original max runtime before filtering for accurate speedup calculation
    original_max_runtime = df['runtime_sec'].max()
    
    valid_df = df[df['runtime_sec'] > 0].copy()
    invalid_df = df[df['runtime_sec'] <= 0].copy()
    
    if len(valid_df) == 0:
        logger.error("No configurations with valid runtime data found!")
        if len(invalid_df) > 0:
            logger.error(f"All {len(invalid_df)} configurations have invalid runtime (likely failed jobs)")
        _log_binary_failure_hints_from_slurm(experiment_dir)
        
        # Save CSV and JSON so that failed jobs are reported
        csv_path = output_dir / 'configuration_scores.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        json_path = output_dir / 'results.json'
        payload = {
            "provenance": build_experiment_provenance(experiment_dir, enable_locality=enable_locality, cpu_only=cpu_only, adaptive_weights=False, weight_mode="static_baseline_with_profile_layout_redistribution"),
            "configurations": df.to_dict('records'),
        }
        json_path.write_text(json.dumps(payload, indent=2))
        return output_dir
    
    if len(invalid_df) > 0:
        logger.warning(f"⚠️  {len(invalid_df)} configurations have invalid runtime (likely failed): {', '.join(invalid_df['config_name'].tolist())}")
        logger.warning(f"   These will be excluded from plots but check SLURM error files for details")
    
    # Use only valid configurations for plots
    df = valid_df
    
    # Speedup metrics: vs fastest (for plots) and vs median (legacy CSV column)
    if len(df) > 0:
        min_runtime = float(df['runtime_sec'].min())
        if min_runtime > 0:
            df['speedup_vs_fastest'] = min_runtime / df['runtime_sec']
            df['slowdown_vs_fastest'] = df['runtime_sec'] / min_runtime
            fastest_name = df.loc[df['runtime_sec'].idxmin(), 'config_name']
            logger.info(
                f"✅ Performance vs fastest ({fastest_name}, {min_runtime:.4f}s): "
                f"1.0 on winner; others are fraction of peak performance"
            )
        else:
            df['speedup_vs_fastest'] = 1.0
            df['slowdown_vs_fastest'] = 1.0

        baseline_runtime = df['runtime_sec'].median()
        if baseline_runtime > 0:
            df['speedup'] = baseline_runtime / df['runtime_sec']
            logger.info(f"✅ Speedup (median baseline {baseline_runtime:.2f}s) kept in CSV as column 'speedup'")
        elif original_max_runtime > 0:
            df['speedup'] = original_max_runtime / df['runtime_sec']
            logger.warning(f"⚠️  Using worst config as speedup baseline (median was 0): {original_max_runtime:.2f}s")
        else:
            df['speedup'] = 1.0
    else:
        logger.warning("⚠️  No valid configurations for speedup calculation")
        df['speedup'] = 1.0
        df['speedup_vs_fastest'] = 1.0
        df['slowdown_vs_fastest'] = 1.0
    
    # Sort by runtime (fastest first) - PRIMARY ranking
    # Heuristic score is useful for efficiency/scalability analysis, but runtime is what matters for performance
    df = df.sort_values('runtime_sec', ascending=True).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Also create a heuristic-based ranking for reference
    df['heuristic_rank'] = df['heuristic_score'].rank(ascending=False, method='min').astype(int)
    
    # Final safeguard before export: pure MPI must have epsilon_openmp = 0.0 (no OpenMP component)
    pure_mpi_export = (df['total_mpi_ranks'] > 1) & (df['omp_threads'] == 1)
    if pure_mpi_export.any():
        df.loc[pure_mpi_export, 'epsilon_openmp'] = 0.0
    
    # Reorder columns to show runtime first (primary), then heuristic score (secondary)
    # Include profiling metrics columns for plotting
    # IMPORTANT: Include profiling columns even if None (for plotting)
    column_order = [
        'rank', 'heuristic_rank', 'config_name', 'mpi_ranks', 'num_nodes', 'total_mpi_ranks', 'omp_threads', 'total_cores',
        'runtime_sec', 'throughput_gflops', 'speedup_vs_fastest', 'slowdown_vs_fastest', 'speedup',
        'heuristic_score', 'heuristic_score_10',
        'alpha_comm', 'alpha_source', 'beta_thread', 'beta_source', 'gamma_locality',
        'delta_gpu', 'delta_gpu_source', 'delta_profile_coverage', 'epsilon_openmp',
        'runtime_score', 'throughput_score', 'efficiency_score', 'scalability_score', 'comm_score',
        'mpi_comm_time', 'mpip_app_time', 'mpip_max_mpi_wall_fraction', 'gpu_busy_time',
        'cpu_utilization', 'openmp_work_efficiency', 'openmp_work_efficiency_display',
        'gpu_utilization', 'gpu_utilization_cupti', 'nvidia_smi_gpu_util', 'profiling_phase_aligned',
        'numa_efficiency', 'locality_source', 'thread_stall_time', 'memory_bandwidth',
        'num_rank_profiles', 'job_id', 'profiling_runtime_used'
    ]
    
    # Ensure profiling columns exist (even if None) - create with 0.0 for plotting
    profiling_cols = ['mpi_comm_time', 'cpu_utilization', 'openmp_work_efficiency', 'gpu_utilization', 'numa_efficiency', 'thread_stall_time']
    for col in profiling_cols:
        if col not in df.columns:
            df[col] = None  # Keep as None to indicate missing data
    # Only include columns that exist in the DataFrame
    available_columns = [col for col in column_order if col in df.columns]
    # Add any remaining columns that weren't in the order
    remaining_cols = [col for col in df.columns if col not in available_columns]
    final_column_order = available_columns + remaining_cols
    df = df[final_column_order]
    
    # Ensure profiling metric columns exist for plotting (convert None to 0.0)
    profiling_cols = ['mpi_comm_time', 'cpu_utilization', 'openmp_work_efficiency', 'gpu_utilization', 'numa_efficiency', 'thread_stall_time']
    for col in profiling_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            # Convert None/NaN to 0.0 for plotting
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    # Save CSV and JSON
    csv_path = output_dir / 'configuration_scores.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Saved scores to: {csv_path}")
    
    json_path = output_dir / 'results.json'
    payload = {
        "provenance": build_experiment_provenance(
            experiment_dir, enable_locality=enable_locality, cpu_only=cpu_only
        ),
        "configurations": df.to_dict('records'),
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Saved JSON to: {json_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    
    # Visualizations are split into standalone modules under figures/
    try:
        plot_auto_tuning_dashboard(df, output_dir, logger, cpu_only=cpu_only)
    except Exception as e:
        logger.error(f"❌ Error generating dashboard plot: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close("all")
        except:
            pass

    try:
        plot_scores_heatmap(df, output_dir, logger, cpu_only=cpu_only)
    except Exception as e:
        logger.error(f"❌ Error generating heatmap plot: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close("all")
        except:
            pass

    try:
        plot_profiling_metrics(df, output_dir, logger)
    except Exception as e:
        logger.error(f"❌ Error generating profiling metrics plot: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close("all")
        except:
            pass

    for _label, _fn in [
        ("rank agreement (runtime vs heuristic)", plot_rank_agreement),
        ("runtime stacked breakdown", plot_runtime_stacked_breakdown),
    ]:
        try:
            _fn(df, output_dir, logger)
        except Exception as e:
            logger.error(f"❌ Error generating {_label}: {e}")
            import traceback
            traceback.print_exc()
            try:
                plt.close("all")
            except Exception:
                pass

    # Print summary - Show fastest config (runtime-based) as primary recommendation
    fastest = df.iloc[0]  # Fastest runtime (primary recommendation)
    
    # Also find best heuristic score config for comparison
    best_heuristic_idx = df['heuristic_score'].idxmax()
    best_heuristic = df.loc[best_heuristic_idx]
    
    # No Nsight SQLite: β/δ from trace unavailable; γ may come from LIKWID; α may come from mpiP
    no_nsys_sqlite = df[df['num_rank_profiles'] == 0]
    if len(no_nsys_sqlite) > 0:
        names = ", ".join(no_nsys_sqlite["config_name"].tolist())
        logger.info(f"\n   Configs with no Nsight SQLite (β/δ from trace unavailable): {names}")
        has_mpip_alpha = no_nsys_sqlite[no_nsys_sqlite["mpi_comm_time"].fillna(0) > 1e-6]
        if len(has_mpip_alpha) > 0:
            logger.info(
                f"   ↑ α (comm) from mpiP for: {', '.join(has_mpip_alpha['config_name'].tolist())}"
            )

    logger.info(
        "\n   Interpretation: Shortest runtime is the primary performance pick; heuristic score "
        "is diagnostic (component balance / efficiency) and may disagree with wall-clock order."
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"⚡ FASTEST CONFIGURATION (Runtime-Based): {fastest['config_name']}")
    logger.info(
        f"   Runtime: {fastest['runtime_sec']:.2f}s | Throughput: {fastest['throughput_gflops']:.2f} "
        f"(GFLOPS or LULESH z/s)"
    )
    _sp = float(fastest.get("speedup_vs_fastest", 1.0) or 1.0)
    logger.info(
        f"   Performance vs fastest: {_sp:.2f}× (1.0 = best) | Heuristic: "
        f"{fastest['heuristic_score_10']:.2f}/10 (raw={fastest['heuristic_score']:.4f})"
    )
    logger.info(f"\n   5 Heuristic Components:")
    logger.info(f"   α Comm: {fastest['alpha_comm']:.3f} | β Thread: {fastest['beta_thread']:.3f} | γ Locality: {fastest['gamma_locality']:.3f}")
    _eps_display = (
        "N/A (pure MPI)"
        if (fastest.get('omp_threads', 0) == 1 and fastest.get('total_mpi_ranks', 0) > 1)
        else f"{fastest['epsilon_openmp']:.3f}"
    )
    logger.info(f"   δ GPU: {fastest['delta_gpu']:.3f} | ε OpenMP: {_eps_display}")
    
    # Show heuristic-based best if different
    if fastest['config_name'] != best_heuristic['config_name']:
        logger.info(f"\n🎯 MOST EFFICIENT CONFIGURATION (Heuristic-Based): {best_heuristic['config_name']}")
        logger.info(
            f"   Runtime: {best_heuristic['runtime_sec']:.2f}s | Heuristic: "
            f"{best_heuristic['heuristic_score_10']:.2f}/10 (raw={best_heuristic['heuristic_score']:.4f})"
        )
        logger.info(f"   Note: Higher heuristic score indicates better efficiency/scalability, but slower runtime.")
        logger.info(f"   Use this if you need multi-node scaling or better resource utilization.")
    
    logger.info(f"\n   Results saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Generate results from auto-tuning experiment')
    parser.add_argument('experiment_dir', type=str, nargs='?', default='latest', help='Path to experiment directory (default: latest)')
    parser.add_argument('--system', type=str, default=None, help='System name (leap2, lonestar6, etc.) for hardware detection')
    parser.add_argument('--enable-likwid', action='store_true', help='Enable LIKWID locality scoring (γ component)')
    parser.add_argument('--update-reports', action='store_true', help='Regenerate auto_tuning_report.md and executive_summary.md from the same data as the CSV (so .md matches comprehensive_results)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    base_dir = REPO_ROOT
    experiments_root = base_dir / "data" / "experiments"

    # Parse experiment directory
    if args.experiment_dir == 'latest':
        possible_roots = experiment_collection_roots(base_dir)
        
        experiment_dir = None
        for root in possible_roots:
            if root.exists():
                try:
                    experiments = [
                        d
                        for d in root.iterdir()
                        if d.is_dir()
                        and d.name.lower().startswith(LATEST_EXPERIMENT_DIR_PREFIXES)
                    ]
                    if experiments:
                        # Sort by modification time
                        experiment_dir = sorted(experiments, key=lambda x: x.stat().st_mtime)[-1]
                        logger.info(f"✅ Using latest experiment: {experiment_dir}")
                        break
                except (PermissionError, OSError) as e:
                    logger.debug(f"Could not access {root}: {e}")
                    continue
        
        if experiment_dir is None:
            logger.error(f"❌ No experiments found. Checked: {possible_roots}")
            logger.error(f"   Make sure you're in the correct directory or provide the experiment path explicitly")
            sys.exit(1)
    else:
        # User provided a path - try multiple locations
        experiment_dir = Path(args.experiment_dir)
        tried_paths = [experiment_dir]
        
        if not experiment_dir.exists():
            # Try as absolute path first
            if not experiment_dir.is_absolute():
                # Try relative to current directory
                experiment_dir = Path.cwd() / args.experiment_dir
                tried_paths.append(experiment_dir)
            if not experiment_dir.exists():
                for root in experiment_collection_roots(base_dir):
                    candidate = root / args.experiment_dir
                    tried_paths.append(candidate)
                    if candidate.exists():
                        experiment_dir = candidate
                        break
            if not experiment_dir.exists():
                experiment_dir = experiments_root / args.experiment_dir
                tried_paths.append(experiment_dir)
            if not experiment_dir.exists():
                experiment_dir = base_dir / args.experiment_dir
                tried_paths.append(experiment_dir)
            if not experiment_dir.exists():
                # Try as absolute path if it looks like one
                if args.experiment_dir.startswith('/'):
                    experiment_dir = Path(args.experiment_dir)
                    tried_paths.append(experiment_dir)
                if not experiment_dir.exists():
                    logger.error(f"❌ Experiment directory not found: {args.experiment_dir}")
                    logger.error(f"   Tried paths:")
                    for p in tried_paths:
                        logger.error(f"     - {p}")
                    sys.exit(1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Regenerating plots for experiment: {experiment_dir}")
    logger.info(f"{'='*60}\n")
    
    # Detect hardware topology to get total cores
    try:
        hardware = detect_hardware_topology(
            system_name=args.system,
            slurm_partition=os.environ.get("HPC_PARTITION"),
        )
        total_cores = hardware.total_cores
        logger.info(f"✅ Detected hardware: {total_cores} cores")
    except Exception as e:
        logger.warning(f"⚠️  Could not detect hardware topology: {e}. Using default 64 cores.")
        total_cores = 64
    
    # Check application profile to determine if locality is enabled
    enable_locality = False
    cpu_only = True
    try:
        if yaml is None:
            raise ImportError("PyYAML not available")
        config_dir = base_dir / "config"
        profile_file = config_dir / "profiles.yaml"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                profiles = yaml.safe_load(f)
            exp_key = yaml_profile_key_for_experiment_dir_name(experiment_dir.name.lower())
            profile = profiles.get(exp_key, profiles.get("sparse_matrix", {}))
            
            enable_locality = profile.get('enable_locality', False)
            cpu_only = profile.get('cpu_only', True)
            # GPU applications: override cpu_only so δ (GPU) score is computed from profiling
            _exp_dir_lower = experiment_dir.name.lower()
            if (
                'gpu' in _exp_dir_lower
                or 'hybrid_vec' in _exp_dir_lower
            ):
                cpu_only = False
                logger.info(
                    "Experiment name suggests a GPU run → "
                    "using GPU profile (cpu_only=False) for δ score"
                )
            logger.info(f"Application profile: locality={enable_locality}, cpu_only={cpu_only}")
    except Exception as e:
        logger.warning(f"Could not read application profile: {e}. Using defaults (locality=False, cpu_only=True)")
    
    # Command-line override
    if args.enable_likwid:
        enable_locality = True
        logger.info(f"🔧 Overriding locality setting from CLI: enable_locality=True")
    
    # Backfill Phase-1 throughput in phase1_results.json from existing slurm-*.out
    if (experiment_dir / "phase1_results.json").exists():
        try:
            backfill_phase1_throughput(experiment_dir)
        except Exception as e:
            logger.debug(f"Phase-1 backfill skipped or failed: {e}")

    # Collect data
    logger.info("📥 Collecting experiment data...")
    all_metrics = collect_experiment_data(experiment_dir, cpu_only=cpu_only)
    cpu_only, all_metrics = apply_cpu_only_when_gpu_absent(experiment_dir, all_metrics, cpu_only)
    
    if not all_metrics:
        logger.error("❌ No metrics collected. Check that:")
        logger.error("   1. SLURM output files (slurm-*.out) exist in the experiment directory")
        logger.error("   2. Profiling data under results/<job_id>/profiling/ or profiling/<job_id>/ (profile.sqlite)")
        logger.error("   3. generated_configurations.json exists")
        sys.exit(1)
    
    if all_metrics:
        logger.info(f"✅ Collected metrics for {len(all_metrics)} configurations:")
        for m in all_metrics:
            logger.info(f"   - {m.get('config_name', 'unknown')}: runtime={m.get('runtime_sec', 0):.2f}s")
        logger.info("")
    else:
        logger.error("❌ No metrics collected - check logs above for details")
    
    # Generate plots
    logger.info("🎨 Generating plots...")
    output_dir = generate_plots(experiment_dir, all_metrics, total_cores, enable_locality, cpu_only)
    
    if output_dir:
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ SUCCESS! Plots generated in: {output_dir}")
        logger.info(f"{'='*60}")
        logger.info(f"   📈 auto_tuning_dashboard.png (with 5 heuristic components)")
        logger.info(f"   🔥 scores_heatmap.png (5 components + composite)")
        logger.info(f"   📊 profiling_metrics.png")
        logger.info(f"   🔀 rank_agreement_runtime_vs_heuristic.png")
        logger.info(f"   📚 runtime_breakdown_stacked.png")
        logger.info(f"   📋 configuration_scores.csv")
        logger.info(f"   📄 results.json")
        logger.info(
            "   (Fastest runtime = primary recommendation; see log block above for heuristic vs time.)"
        )
        logger.info(f"{'='*60}\n")
        # Optionally regenerate .md reports so they match the CSV (Comm Score, best config, etc.)
        if getattr(args, 'update_reports', False):
            try:
                from autotuner.utils.report_generator import AutoTuningReportGenerator
                csv_path = output_dir / "configuration_scores.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    all_results = {}
                    for _, row in df.iterrows():
                        name = str(row.get('config_name', ''))
                        all_results[name] = {
                            'score': float(row.get('heuristic_score', 0)),
                            'heuristic_score_10': float(
                                row.get('heuristic_score_10', float(row.get('heuristic_score', 0)) * 10.0)
                            ),
                            'mpi_ranks': int(row.get('mpi_ranks', 0)),
                            'omp_threads': int(row.get('omp_threads', 0)),
                            'performance_metrics': {
                                'alpha_comm': float(row.get('alpha_comm', 0)),
                                'comm_score': float(row.get('comm_score', 0)),
                                'cpu_utilization': float(row.get('cpu_utilization', 0)),
                                'numa_efficiency': float(row.get('numa_efficiency', 0)),
                            }
                        }
                    best_row = df.loc[df['heuristic_score'].idxmax()]
                    experiment_data = {
                        'experiment_id': experiment_dir.name,
                        'application': experiment_dir.name.split('_')[0] if '_' in experiment_dir.name else experiment_dir.name,
                        'timestamp': time.time(),
                        'hardware_info': {'total_cores': total_cores, 'system_name': 'HPC System', 'sockets': 2, 'numa_domains': 2},
                        'configurations_tested': len(df),
                        'best_configuration': {
                            'name': str(best_row['config_name']),
                            'score': float(best_row['heuristic_score']),
                            'details': all_results.get(str(best_row['config_name']), {}),
                        },
                        'all_results': all_results,
                    }
                    gen = AutoTuningReportGenerator()
                    report_file = gen.generate_report(experiment_data, experiment_dir, scorer=None)
                    logger.info(f"   📄 Regenerated reports: auto_tuning_report.md, executive_summary.md")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not regenerate .md reports: {e}")
    else:
        logger.error("❌ Failed to generate plots - generate_plots() returned None")
        logger.error("   Check the logs above for specific errors")
        sys.exit(1)


if __name__ == '__main__':
    main()

