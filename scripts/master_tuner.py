#!/usr/bin/env python3
"""Master auto-tuner: generate configs, run SLURM jobs, score, and report."""

import sys
import argparse
import logging
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import json
import os
import re
import subprocess
import shutil
import platform
import warnings

# Import for visualization (installed via pip)
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Repository root on sys.path; scripts/ added only for generate_results.py imports
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))  # generate_results.py (not under autotuner/)

from autotuner.core.heuristic_scoring import ConfigurationScorer, AdaptiveConfigurationScorer, PerformanceMetrics
from autotuner.core.configuration_generator import ConfigurationGenerator, detect_hardware_topology
from autotuner.core.metrics_extractor import NsightMetricsExtractor
from autotuner.core.phase1_subset_selector import collect_phase1_data, select_subset
from autotuner.automation.slurm_manager import SlurmManager, SLURMJobConfig
from autotuner.utils.report_generator import AutoTuningReportGenerator
from autotuner.utils.path_utils import (
    find_slurm_log_for_job,
    get_logical_cwd,
    normalize_to_logical_path,
    read_slurm_log_excerpt,
    slurm_log_candidate_paths,
)

def _plt_tight_layout() -> None:
    if not VISUALIZATION_AVAILABLE:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*[Tt]ight layout not applied.*", category=UserWarning)
        plt.tight_layout()


from autotuner.app_registry import (
    APPLICATION_PROFILES,
    resolve_profile_key,
    resolve_default_executable_path,
    build_configurations_for_application,
    effective_mpip_for_job,
    likwid_rank0_only_for_job,
    phase1_likwid_time_warning,
    apply_application_cli_defaults,
    infer_stdout_metadata,
    parse_job_stdout_runtime_throughput,
    parse_slurm_file_runtime_throughput,
)
from autotuner.app_registry.lulesh import (
    reference_mpi_rank_count_from_configs,
    scale_arguments_for_equal_total_zones,
)
from autotuner.app_registry.minimd import merge_minimd_phase1_cli

# Configure logging with both file and console output
def setup_logging(log_dir: Path = None, verbose: bool = False):
    """Setup logging to file and console"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # File handler (if log_dir specified)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"master_tuner_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always capture everything to file
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        print(f"📝 Logging to: {log_file}")
    
    # Also log to master_tuner.log in CWD (User Request)
    try:
        cwd_log = Path("master_tuner.log")
        cwd_handler = logging.FileHandler(cwd_log, mode='a') # Append mode
        cwd_handler.setLevel(logging.DEBUG)
        cwd_handler.setFormatter(formatter)
        handlers.append(cwd_handler)
        print(f"📝 Also logging to: {cwd_log.absolute()}")
    except Exception as e:
        print(f"⚠️ Could not log to master_tuner.log: {e}")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

# Initial logger (will be reconfigured when work_directory is known)
logger = logging.getLogger(__name__)


def _env_path_has_executable_nsys(env_path: str) -> bool:
    """True if any colon-separated directory in env_path contains an executable ``nsys``."""
    if not env_path or not str(env_path).strip():
        return False
    for part in str(env_path).split(":"):
        base = os.path.expanduser(part.strip())
        if not base:
            continue
        candidate = os.path.join(base, "nsys")
        try:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return True
        except OSError:
            continue
    return False


def _reject_non_linux_executable_for_hpc(executable: Path) -> None:
    """
    If ``file(1)`` is available, reject macOS Mach-O binaries (and similar) before any SLURM work.
    Login-node ``exists`` checks do not catch wrong-OS builds on a shared filesystem.
    """
    fu = shutil.which("file")
    if not fu or not executable.is_file():
        return
    try:
        cp = subprocess.run(
            [fu, str(executable)],
            capture_output=True,
            text=True,
            timeout=8,
        )
        out = ((cp.stdout or "") + (cp.stderr or "")).lower()
    except (OSError, subprocess.TimeoutExpired):
        return
    if "mach-o" in out:
        raise RuntimeError(
            f"\n\n❌  Executable is macOS (Mach-O), not Linux ELF: {executable}\n"
            f"    HPC GPU nodes cannot run it (Exec format error).\n\n"
            f"    Fix: on the cluster, build for Linux x86_64, e.g.\n"
            f"      cd applications && make hybrid_vec_gpu\n"
            f"    Then pass: --executable \"$PWD/applications/HybridVec/hybrid_vec_gpu\"\n"
        )
    if "pe32" in out or "pe32+" in out or "ms windows" in out:
        raise RuntimeError(
            f"\n\n❌  Executable appears to be a Windows PE file: {executable}\n"
            f"    Build on the cluster: cd applications && make hybrid_vec_gpu\n"
        )
    # x86_64 login nodes cannot run AArch64 ELF (and vice versa).
    if platform.system().lower() == "linux" and platform.machine().lower() in ("x86_64", "amd64"):
        if "elf" in out and ("aarch64" in out or "arm64" in out):
            raise RuntimeError(
                f"\n\n❌  Executable is Linux ARM64 (AArch64) ELF but this host is x86_64: {executable}\n"
                f"    Rebuild for x86_64: cd applications && make hybrid_vec_gpu\n"
            )


# Default prescreen (Phase-1) args when --phase1-args is not set: fixed size and iterations.
PHASE1_DEFAULT_ARGS = "--size 4096 --iterations 2"

# Cap --size when deriving Phase-1 / default Phase-2 CLI from full --arguments (per-app tuning policy).
GPU_MPI_LIGHT_SIZE_CAP = 512


def _phase1_lightweight_arguments(
    arguments: str,
    phase1_iterations: int = 2,
    phase1_size_cap: Optional[int] = 4096,
) -> str:
    """Reduce application arguments for Phase-1 when deriving from full args (e.g. if caller wants cap-only).
    - Replaces --iterations N with --iterations phase1_iterations (default 2).
    - If phase1_size_cap is set, replaces --size N with min(N, phase1_size_cap) (default 4096).
    """
    if not arguments or not arguments.strip():
        return arguments
    out = re.sub(r'--iterations\s+\d+', f'--iterations {phase1_iterations}', arguments)
    if phase1_size_cap is not None:
        def _cap_size(m):
            n = int(m.group(1))
            return f'--size {min(n, phase1_size_cap)}'
        out = re.sub(r'--size\s+(\d+)', _cap_size, out)
    return out


class MasterAutoTuner:
    """
    Master orchestrator for MPI+OpenMP auto-tuning
    
    Implements the complete workflow described in the research paper:
    - Hardware topology detection
    - Configuration generation
    - Automated job execution
    - Performance analysis
    - Heuristic scoring
    - Result reporting
    """
    
    def __init__(self, 
                 application_name: str, 
                 account: str,
                 system_name: str = None,
                 work_directory: Optional[Path] = None,
                 adaptive_weights: bool = False,
                 enable_likwid: bool = False,
                 likwid_path: Optional[str] = None,
                 enable_mpip: bool = True,
                 mpip_path: Optional[str] = None,
                 no_mpip_srun_run1a: bool = False,
                 cpu_only: bool = True,
                 num_nodes: int = 1,
                 profile_subset: bool = False,
                 skip_phase1: bool = False,
                 no_nsight: bool = False,
                 multi_rank_profile: bool = True,  # Enabled by default for full coverage
                 nsight_duration_sec: Optional[int] = None,
                 nsight_trace_domains: Optional[str] = None,
                 phase1_time_limit: Optional[str] = None,
                 phase2_time_limit: Optional[str] = None,
                 env_path: Optional[str] = None,
                 deep_profile_finalists: bool = False,
                 deep_profile_max_configs: int = 3,
                 deep_profile_time_limit: Optional[str] = None,
                 phase2_all_configs: bool = False):
        """
        Initialize the master auto-tuner
        
        Args:
            application_name: Name of the application to tune 
            account: HPC allocation account
            system_name: HPC system name
            work_directory: Working directory for experiments
            adaptive_weights: If True, use adaptive heuristic weights (rule-based, no ML/AI)
            enable_likwid: If True, enable LIKWID profiling for memory locality
            likwid_path: Optional path to LIKWID install (e.g. $HOME/software/likwid). If set, job script exports LIKWID_PATH so compute nodes find LIKWID.
            enable_mpip: If True, job script tries to load mpiP for α (communication) from PMPI-level timing. Requires libmpiP.so in mpip_path or default locations.
            mpip_path: Optional path to mpiP install (e.g. $HOME/software/mpiP). If set, job script checks this path first for libmpiP.so.
            no_mpip_srun_run1a: If True, disable MVAPICH2 Run-1a mpiP via srun+LD_PRELOAD (use plain mpirun; no mpiP preload on Hydra).
            cpu_only: If True, redistribute GPU/OpenMP weights to comm/thread (for CPU-only workloads)
            profile_subset: If True, use two-phase: Phase-1 preliminary screening (runtime, throughput, LIKWID), then full profiling only for selected subset.
            skip_phase1: If True (and profile_subset True), skip Phase-1 and run full profiling for all configs (use when Phase-1 times out).
            no_nsight: If True, do not use Nsight; run application only and score from runtime/throughput (fast, hours not days).
            multi_rank_profile: If True, multi-rank jobs run Run 2 (Nsight wraps full mpirun) after Run 1; if False, multi-rank is Run 1 only (no Nsight).
            nsight_duration_sec: If set, limit Nsight trace to this many seconds (faster profiling, α/β from slice; e.g. 60 or 120).
            nsight_trace_domains: Comma list for nsys --trace. Default osrt,mpi,cuda,cublas (GPU jobs); CPU-only default is osrt,mpi. Use --nsight-trace to override.
            phase1_time_limit: Time limit for Phase-1 jobs (default 00:20:00 when profile_subset; override with e.g. 00:15:00).
            phase2_time_limit: Time limit for profiling jobs (Nsight+LIKWID Run 2). If unset, uses the same as run_complete_auto_tuning time_limit.
                Full multi-node Nsight often needs >02:00:00; set e.g. 06:00:00 to avoid TIME LIMIT during Run 2.
            env_path: Additional directories to prepend to PATH in job scripts (colon-separated).
            deep_profile_finalists: If True, after the main profiling batch finishes, pick up to
                deep_profile_max_configs finalists (fastest runtime, best heuristic score, median runtime)
                and submit extra jobs with Nsight on representative MPI ranks {0, N/2, N-1}.
            deep_profile_max_configs: Max finalist configs for the deep spread phase (default 3).
            deep_profile_time_limit: Optional per-job time limit for deep jobs (default: same as main time_limit).
            phase2_all_configs: With profile_subset: if True, Phase-2 profiles every config (even Phase-1 failures).
                If False (default), Phase-2 only runs the heuristic subset of configs that completed Phase-1 with a parsable runtime.
        """
        self.application_name = application_name
        self.account = account
        self.system_name = system_name
        self.adaptive_weights = adaptive_weights
        self.enable_likwid = enable_likwid
        self.likwid_path = likwid_path
        self.enable_mpip = enable_mpip
        self.mpip_path = mpip_path
        self.no_mpip_srun_run1a = no_mpip_srun_run1a
        self.cpu_only = cpu_only
        self.num_nodes = num_nodes
        self.profile_subset = profile_subset
        self.skip_phase1 = skip_phase1
        self.no_nsight = no_nsight
        self.multi_rank_profile = multi_rank_profile
        self.nsight_duration_sec = nsight_duration_sec
        # CPU-only jobs on nodes without GPUs: cuda,cublas tracing often yields empty/failed exports.
        self.nsight_trace_domains = nsight_trace_domains
        if self.cpu_only and not (self.nsight_trace_domains and str(self.nsight_trace_domains).strip()):
            self.nsight_trace_domains = "osrt,mpi"
        self.phase1_time_limit = phase1_time_limit
        if phase2_time_limit is not None and str(phase2_time_limit).strip():
            self.phase2_time_limit = str(phase2_time_limit).strip()
        else:
            self.phase2_time_limit = None
        self.env_path = env_path
        self.deep_profile_finalists = deep_profile_finalists
        self.deep_profile_max_configs = max(1, int(deep_profile_max_configs))
        self.deep_profile_time_limit = deep_profile_time_limit
        self.phase2_all_configs = bool(phase2_all_configs)
        # When GPU-mode scoring sees ~zero GPU in every trace (broken binary, no kernels), we
        # switch to CPU-only weight layout so α–β–γ dominate like sparse_matrix (see _maybe_use_cpu_weight_scoring_when_gpu_absent).
        self._gpu_scoring_fallback = False

        # Set up working directory
        if work_directory is None:
            timestamp = int(time.time())
            self.work_directory = Path(f"data/experiments/{application_name}_{timestamp}")
        else:
            self.work_directory = Path(work_directory)
        
        self.work_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to experiment's logs folder
        global logger
        log_dir = self.work_directory / "logs"
        logger = setup_logging(log_dir=log_dir, verbose=False)
        
        # Initialize components
        logger.info(f"=== Auto-Tuning Session Started ===")
        logger.info(f"Experiment directory: {self.work_directory}")
        logger.info("Initializing auto-tuning components...")
        
        # Hardware detection (use system_name for accurate TACC specs)
        self.hardware = detect_hardware_topology(
            system_name=self.system_name,
            slurm_partition=os.environ.get("HPC_PARTITION"),
        )
        logger.info(f"Detected hardware: {self.hardware}")
        
        # Configuration generator
        self.config_generator = ConfigurationGenerator(self.hardware)
        
        # Initialize SLURM Manager — must pass system_name so job scripts load hpc_config.json modules
        _sys = (self.system_name or "default").strip().lower() or "default"
        self.slurm_manager = SlurmManager(self.work_directory, system_name=_sys)
        _mods = self.slurm_manager.system_config.get("modules") or []
        logger.info(
            "SLURM manager system=%r (%d module(s) in job scripts)%s",
            self.slurm_manager.system_name,
            len(_mods),
            f": {_mods}" if _mods else " — add config/hpc_config.json (see config/hpc_config.example.json) if modules should load on compute nodes",
        )
        _excl = (self.slurm_manager.system_config.get("sbatch_exclude") or "").strip()
        if _excl:
            logger.info(
                "GPU/SLURM node exclude active: %s — jobs will not run on these nodes (PENDING until other gpu1 nodes are free)",
                _excl,
            )
        if not _mods and not self.cpu_only:
            logger.warning(
                "GPU / hybrid runs with **zero** modules in SLURM scripts. Fix: place config/hpc_config.json "
                "in the repo (or ~/.config/autotuner/), define modules (e.g. gcc, cuda, mvapich2-gdr), and pass "
                "--system <key> (e.g. --system ls6)."
            )
        
        # Initialize other componentsr (adaptive or standard)
        # CPU-only mode: redistributes GPU and OpenMP weights to comm and thread
        if adaptive_weights:
            self.scorer = AdaptiveConfigurationScorer(enable_adaptation=True, enable_locality=enable_likwid, cpu_only=cpu_only)
            logger.info(f"Using adaptive heuristic weights ({'CPU-only' if cpu_only else 'Full'} mode, locality {'enabled' if enable_likwid else 'disabled'})")
        else:
            self.scorer = ConfigurationScorer(enable_locality=enable_likwid, cpu_only=cpu_only)
            logger.info(f"Using fixed heuristic weights ({'CPU-only' if cpu_only else 'Full'} mode, locality {'enabled' if enable_likwid else 'disabled'})")
        
        # Report generator
        self.report_generator = AutoTuningReportGenerator()
        
        # Experiment tracking
        self.experiment_id = f"autotune_{application_name}_{int(time.time())}"
        self.experiment_results = {
            'experiment_id': self.experiment_id,
            'application': application_name,
            'timestamp': time.time(),
            'hardware_info': {
                'total_cores': self.hardware.total_cores,
                'sockets': self.hardware.sockets,
                'numa_domains': self.hardware.numa_domains,
                'system_name': self.hardware.system_name
            },
            'configurations_tested': 0,
            'successful_runs': 0,
            'best_configuration': None,
            'all_results': {}
        }
        
        logger.info(f"Initialized Master Auto-Tuner for {application_name}")
        logger.info(f"Working directory: {self.work_directory}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Using MPI (Message Passing Interface) communication backend")
    
    def run_complete_auto_tuning(self, 
                                executable_path: str,
                                arguments: str = "",
                                time_limit: str = "02:00:00",
                                pilot_only: bool = False,
                                phase1_arguments: Optional[str] = None,
                                phase2_arguments: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete auto-tuning workflow
        
        Args:
            executable_path: Path to the application executable
            arguments: Command line arguments for the application
            time_limit: Time limit for each job
            pilot_only: If True, run only a subset of configurations for testing
            
        Returns:
            Dictionary with auto-tuning results
        """
        logger.info("🚀 Starting complete auto-tuning workflow")
        logger.info("=" * 60)

        # Phase-2 / full-profiling CLI: optional override; hybrid_vec defaults to workload-safe cap.
        self._phase2_arguments_cli = phase2_arguments
        self._logged_profiling_cli = False

        # ── Executable pre-flight check ────────────────────────────────────────
        # Validate NOW on the login node (shared FS = same view as compute nodes).
        # This prevents wasting cluster allocation on jobs that will all fail with
        # "No such file or directory" before a single instruction runs.
        _exe_raw = executable_path
        _exe_expanded = Path(os.path.expandvars(os.path.expanduser(_exe_raw)))
        if not _exe_expanded.exists() or not _exe_expanded.is_file():
            # Build a helpful suggestion: search siblings and subdirs for similarly-named files
            _parent = _exe_expanded.parent
            _stem = _exe_expanded.stem  # e.g. "hybrid_vec_gpu"
            _candidates = []
            for _search_root in [_parent, _parent.parent]:
                if _search_root.exists():
                    for _f in sorted(_search_root.rglob("*")):
                        if _f.is_file() and _stem.split("_")[0] in _f.name.lower() and os.access(_f, os.X_OK):
                            _candidates.append(str(_f))
            _hint = ""
            if _candidates:
                _hint = "\n  Nearby executables found:\n" + "\n".join(f"    {c}" for c in _candidates[:5])
            raise FileNotFoundError(
                f"\n\n❌  Executable not found: {_exe_expanded}\n"
                f"    (Raw path given: {_exe_raw})\n"
                f"    Jobs would fail on every compute node before running a single instruction."
                + _hint +
                f"\n\n  Fix: pass the correct path with --executable, e.g.:\n"
                f"    --executable \"{_candidates[0]}\"\n" if _candidates else
                f"\n\n  Fix: build the application first, then pass the correct path with --executable.\n"
            )
        logger.info(f"  ✓ Executable validated: {_exe_expanded}")
        _reject_non_linux_executable_for_hpc(_exe_expanded)
        # ──────────────────────────────────────────────────────────────────────

        if self.no_nsight:
            logger.info("⚡ Fast mode (--no-nsight): application only, scoring from runtime/throughput; no Nsight profiling.")
        else:
            _nt = getattr(self, "nsight_trace_domains", None) or "osrt,mpi,cuda,cublas,nvtx"
            logger.info(
                f"Nsight trace: {_nt}. Metrics: α (comm), β (thread), γ (LIKWID), δ (GPU); "
                "ε (OpenMP) from trace scheduling/CPU analysis (see metrics_extractor)."
            )
        if getattr(self, 'enable_mpip', True):
            logger.info("mpiP enabled for α (communication) when libmpiP.so is found (PMPI-level timing).")
            _sc = getattr(self.slurm_manager, "system_config", None) or {}
            _mods_s = " ".join(_sc.get("modules") or []).lower()
            _mvapich = "mvapich" in _mods_s
            # Align with job_generator: missing key → enable srun preload (same as additional_options default).
            _raw_mv = _sc.get("mpip_mvapich_srun")
            _srun_mpip = True if _raw_mv is None else bool(_raw_mv)
            _no_srun = bool(getattr(self, "no_mpip_srun_run1a", False))
            if _mvapich and _srun_mpip and not _no_srun:
                logger.info(
                    "  mpiP: MVAPICH2 — Run 1a uses srun+LD_PRELOAD (Hydra-safe) when libmpiP.so is found "
                    "(hpc_config mpip_mvapich_srun). Use --no-mpip-srun-run1a if srun fails on your site."
                )
            elif _mvapich and (not _srun_mpip or _no_srun):
                logger.info(
                    "  mpiP: MVAPICH2 — Run 1a mpiP preload disabled (mpip_mvapich_srun=false or --no-mpip-srun-run1a); "
                    "α from Nsight osrt,mpi unless you enable srun preload in hpc_config."
                )
            else:
                logger.info(
                    "  mpiP: OpenMPI-style — Run 1a uses mpirun -x LD_PRELOAD when libmpiP.so is found."
                )
            logger.info("  If no .mpiP files appear: set --mpip-path to the mpiP install visible on compute nodes (e.g. $HOME/software/mpiP with lib/libmpiP.so); α can still come from mpiP stdout or Nsight.")
            # Validate mpiP path so α is instrumented (4 perfect metrics): warn if lib not found
            mpip_path = getattr(self, 'mpip_path', None)
            if mpip_path:
                expanded = Path(os.path.expanduser(mpip_path))
                lib_candidates = [expanded / "lib" / "libmpiP.so", expanded / "libmpiP.so", expanded / "lib" / "libmpiP.a"]
                found = any(p.exists() for p in lib_candidates)
                if not found:
                    logger.warning("  ⚠️  --mpip-path does not contain libmpiP.so (checked lib/libmpiP.so, libmpiP.so). α may fall back to Nsight or be zero.")
                elif _mvapich and _srun_mpip and not _no_srun:
                    logger.info("  ✓ mpiP path validated; MVAPICH2 Run 1a uses srun+LD_PRELOAD for PMPI when the library is on compute nodes.")
                elif _mvapich:
                    logger.info("  ✓ mpiP path validated; MVAPICH2 preload is off in config — enable mpip_mvapich_srun in hpc_config for PMPI reports.")
                else:
                    logger.info("  ✓ mpiP path validated (lib found); OpenMPI Run 1a preloads mpiP via -x LD_PRELOAD.")
            else:
                logger.info("  Set --mpip-path to the mpiP install on compute nodes for reliable α (PMPI).")
        if getattr(self, 'enable_likwid', False):
            likwid_path = getattr(self, 'likwid_path', None)
            if likwid_path:
                expanded = Path(os.path.expanduser(likwid_path))
                perfctr = expanded / "bin" / "likwid-perfctr"
                if not perfctr.exists():
                    perfctr = expanded / "likwid-perfctr"
                if not expanded.exists() or not perfctr.exists():
                    logger.warning("  ⚠️  --likwid-path not found or missing bin/likwid-perfctr. γ (locality) may use fallback or be zero.")
                else:
                    logger.info("  ✓ LIKWID path validated; γ will use hardware counters when jobs run.")
            else:
                logger.warning("  ⚠️  --enable-likwid set but --likwid-path not set; γ may be unavailable on compute nodes.")
        
        try:
            # Step 1: Generate configurations
            logger.info("Step 1: Generating MPI+OpenMP configurations...")
            configurations = self._generate_configurations(pilot_only)
            
            # Step 2: Execute applications (single-phase or two-phase "profiling only a subset")
            if self.profile_subset and not self.skip_phase1:
                logger.info("Step 2a: Phase-1 Preliminary Screening (runtime, throughput, LIKWID only)...")
                # Default Phase-1 time limit to 20 min (queue + startup can eat 10 min); override with --phase1-time-limit
                phase1_time = self.phase1_time_limit or "00:20:00"
                # Warn if very short: Phase-1 jobs can hit TIME LIMIT (one config may be slow)
                if self.phase1_time_limit:
                    parts = self.phase1_time_limit.strip().split(":")
                    mins = int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)
                    if mins < 10:
                        logger.warning("  ⚠️  Phase-1 time limit is under 10 minutes; slow configs may be CANCELLED (TIME LIMIT). Consider --phase1-time-limit 00:10:00 or 00:20:00.")
                    p1w = phase1_likwid_time_warning(
                        self.application_name,
                        getattr(self, "enable_likwid", False),
                        self.phase1_time_limit,
                    )
                    if p1w:
                        logger.warning(p1w)
                # Prescreen default: scale down user arguments (iterations=2) to prevent OOM
                if phase1_arguments is None:
                    if self.application_name == "hybrid_vec" and not getattr(
                        self, "cpu_only", True
                    ):
                        # Cap Phase-1 --size (VRAM / contention); Phase-2 defaults match unless --phase2-args.
                        phase1_arguments = _phase1_lightweight_arguments(
                            arguments, phase1_size_cap=GPU_MPI_LIGHT_SIZE_CAP
                        )
                    else:
                        phase1_arguments = _phase1_lightweight_arguments(arguments)
                if self.application_name == "minimd":
                    _p1_before = (phase1_arguments or "").strip()
                    phase1_arguments = merge_minimd_phase1_cli(arguments, phase1_arguments or "")
                    if (phase1_arguments or "").strip() != _p1_before:
                        logger.info(
                            "Phase-1 miniMD: merged -i/--input_file from full --arguments into "
                            "prescreen CLI (--phase1-args alone omits the input deck)."
                        )
                if phase1_arguments != arguments:
                    logger.info(f"Phase-1 using lightweight arguments: {phase1_arguments.strip()!r}")
                logger.info(f"Phase-1 time limit per job: {phase1_time}")
                job_results_phase1 = self._execute_applications(
                    configurations, executable_path, phase1_arguments, phase1_time, profiling=False
                )
                config_map = {c.name: c for c in configurations}
                phase1_data = collect_phase1_data(
                    self.work_directory,
                    job_results_phase1,
                    config_map,
                    system_config=getattr(self.slurm_manager, "system_config", None),
                )
                missing_p1 = [c.name for c in configurations if c.name not in phase1_data]
                if missing_p1:
                    logger.warning(
                        "Phase-1 has no parsable runtime for %d config(s) (FAILED job or missing logs): %s",
                        len(missing_p1),
                        missing_p1,
                    )
                if not phase1_data:
                    if self._all_job_logs_show_binary_exec_failure(job_results_phase1):
                        msg = (
                            "Phase-1 logs show Exec format error / cannot execute binary file for the "
                            "application on compute nodes. The executable is not runnable on this partition "
                            "(wrong CPU architecture or incompatible build). Rebuild on the cluster with the "
                            "same modules as the job (e.g. gpu1 + gcc/MVAPICH2/CUDA) before re-running. "
                            "Aborting — skipping Phase-2 full profiling to avoid wasted GPU hours."
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)
                    _sc = getattr(self.slurm_manager, "system_config", None) or {}
                    _log_hint = _sc.get("slurm_log_dir") or "/scratch/$USER"
                    self._log_phase1_failure_diagnostics(job_results_phase1, _sc)
                    msg = (
                        "Phase-1 produced no valid runtime for any configuration (all jobs FAILED or "
                        "logs not collected). Inspect SLURM stderr (e.g. "
                        f"{_log_hint}/slurm-<jobid>.err). "
                        "Aborting — not launching Phase-2 full profiling (avoids wasted GPU hours). "
                        "Fix the batch failure, sync code, and re-run."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
                else:
                    if self.phase2_all_configs:
                        selected_names = [c.name for c in configurations]
                        selected_configs = list(configurations)
                        logger.info(
                            "Phase-2: profiling all %d configs (--phase2-all-configs); Phase-1 data for reporting only.",
                            len(selected_configs),
                        )
                    else:
                        selected_names = select_subset(phase1_data, configurations)
                        selected_configs = [c for c in configurations if c.name in selected_names]
                        skipped_p2 = [c.name for c in configurations if c.name not in selected_names]
                        logger.info(
                            "Phase-1 subset selection: %d config(s) for Phase-2: %s",
                            len(selected_configs),
                            selected_names,
                        )
                        if skipped_p2:
                            logger.info(
                                "Skipping Phase-2 for %d config(s) (subset mode; use --phase2-all-configs to profile all): %s",
                                len(skipped_p2),
                                skipped_p2,
                            )
                    self.experiment_results["phase1_configs"] = len(configurations)
                    self.experiment_results["phase1_selected"] = selected_names
                    self.experiment_results["phase1_data"] = {
                        k: {
                            "runtime": v["runtime"],
                            "throughput": v["throughput"],
                            "locality": v.get("locality", v.get("locality_estimate", 0)),
                        }
                        for k, v in phase1_data.items()
                    }
                    phase1_summary = {
                        "phase1_configs": len(configurations),
                        "phase1_selected": selected_names,
                        "phase1_completed_with_metrics": sorted(phase1_data.keys()),
                        "phase1_data": self.experiment_results["phase1_data"],
                    }
                    phase1_file = self.work_directory / "phase1_results.json"
                    with open(phase1_file, "w") as f:
                        json.dump(phase1_summary, f, indent=2)
                    logger.info(f"Phase-1 summary saved to {phase1_file}")
                    logger.info("Step 2b: Phase-2 full profiling (Nsight + LIKWID) for selected subset...")
                    p2_tl = self._time_limit_for_profiling(time_limit)
                    logger.info(f"Phase-2 SLURM time limit per job: {p2_tl}")
                    if not self.no_nsight and not self.phase2_time_limit:
                        logger.info(
                            "  If Run 2 (Nsight) is CANCELLED for TIME LIMIT, increase wall time, e.g. --phase2-time-limit 06:00:00 "
                            "(multi-node full traces + export often exceed 02:00:00)."
                        )
                    job_results = self._execute_applications(
                        selected_configs,
                        executable_path,
                        self._resolve_profiling_arguments(arguments),
                        p2_tl,
                        profiling=not self.no_nsight,
                    )
                    phase2_job_ids = [str(r["job_id"]) for r in job_results.values() if "job_id" in r]
                    phase2_job_ids_file = self.work_directory / "phase2_job_ids.json"
                    with open(phase2_job_ids_file, "w") as f:
                        json.dump(phase2_job_ids, f)
                    logger.info(f"Phase-2 job IDs saved to {phase2_job_ids_file} (for generate_results)")
            elif self.profile_subset and self.skip_phase1:
                logger.info("Step 2: Skipping Phase-1 (--skip-phase1); full profiling for all configs...")
                prof_tl = self._time_limit_for_profiling(time_limit) if not self.no_nsight else time_limit
                job_results = self._execute_applications(
                    configurations,
                    executable_path,
                    self._resolve_profiling_arguments(arguments),
                    prof_tl,
                    profiling=not self.no_nsight,
                )
                # Save job IDs for generate_results (same as Phase 2)
                all_job_ids = [str(r["job_id"]) for r in job_results.values() if "job_id" in r]
                job_ids_file = self.work_directory / "phase2_job_ids.json"
                with open(job_ids_file, "w") as f:
                    json.dump(all_job_ids, f)
                logger.info(f"Job IDs saved to {job_ids_file} (for generate_results)")
            else:
                logger.info("Step 2: Executing application runs (full profiling for all configs)...")
                prof_tl = self._time_limit_for_profiling(time_limit) if not self.no_nsight else time_limit
                job_results = self._execute_applications(
                    configurations,
                    executable_path,
                    self._resolve_profiling_arguments(arguments),
                    prof_tl,
                    profiling=not self.no_nsight,
                )
                # Save job IDs for generate_results
                all_job_ids = [str(r["job_id"]) for r in job_results.values() if "job_id" in r]
                job_ids_file = self.work_directory / "phase2_job_ids.json"
                with open(job_ids_file, "w") as f:
                    json.dump(all_job_ids, f)
                logger.info(f"Job IDs saved to {job_ids_file} (for generate_results)")
            
            # Persist configurations into experiment results for downstream scoring lookup
            try:
                self.experiment_results['all_results'] = {
                    name: {'config': data.get('config')} for name, data in job_results.items()
                    if isinstance(data, dict) and data.get('config') is not None
                }
            except Exception as e:
                logger.warning(f"Could not persist job configuration mapping for scoring: {e}")

            job_results = self._maybe_run_deep_profile_finalists(
                job_results, executable_path, arguments, time_limit
            )
            try:
                self.experiment_results['all_results'] = {
                    name: {'config': data.get('config')} for name, data in job_results.items()
                    if isinstance(data, dict) and data.get('config') is not None
                }
            except Exception as e:
                logger.warning(f"Could not refresh job configuration mapping after deep phase: {e}")
            
            # Step 3: Extract and analyze performance metrics
            logger.info("Step 3: Extracting and analyzing performance metrics...")
            performance_data = self._extract_performance_metrics(job_results)
            self._maybe_use_cpu_weight_scoring_when_gpu_absent(performance_data)

            # Step 4: Score configurations using heuristic function
            logger.info("Step 4: Scoring configurations using heuristic function...")
            scored_configs = self._score_configurations(performance_data)
            
            # Step 4b: Adaptive weight adjustment (if enabled)
            if self.adaptive_weights and isinstance(self.scorer, AdaptiveConfigurationScorer):
                logger.info("Step 4b: Adapting heuristic weights based on performance analysis...")
                try:
                    # Adapt weights based on all results
                    adapted_weights = self.scorer.adapt_weights_from_results(scored_configs)
                    
                    # Re-score with adapted weights
                    logger.info("Re-scoring configurations with adapted weights...")
                    scored_configs = self._score_configurations(performance_data)
                    
                    # Store adaptation history
                    adaptation_history = self.scorer.get_adaptation_history()
                    if adaptation_history:
                        self.experiment_results['weight_adaptation'] = {
                            'initial_weights': {
                                'alpha': self.scorer.initial_weights['alpha'],
                                'beta': self.scorer.initial_weights['beta'],
                                'gamma': self.scorer.initial_weights['gamma'],
                                'delta': self.scorer.initial_weights['delta'],
                                'epsilon': self.scorer.initial_weights['epsilon']
                            },
                            'final_weights': {
                                'alpha': self.scorer.alpha,
                                'beta': self.scorer.beta,
                                'gamma': self.scorer.gamma,
                                'delta': self.scorer.delta,
                                'epsilon': self.scorer.epsilon
                            },
                            'adaptation_history': adaptation_history
                        }
                        logger.info(f"✅ Weight adaptation completed. Final weights: "
                                   f"α={self.scorer.alpha:.3f}, β={self.scorer.beta:.3f}, "
                                   f"γ={self.scorer.gamma:.3f}, δ={self.scorer.delta:.3f}, "
                                   f"ε={self.scorer.epsilon:.3f}")
                except Exception as e:
                    logger.warning(f"Weight adaptation failed (non-critical): {e}")
                    logger.info("Continuing with original weights...")
            
            # Step 5: Generate comprehensive report
            logger.info("Step 5: Generating comprehensive report...")
            report_path = self._generate_final_report(scored_configs)
            
            # Note: Enhanced visualization is now integrated into comprehensive_results (Step 7)

            
            # Step 6: Provide recommendations
            logger.info("Step 6: Providing optimization recommendations...")
            recommendations = self._generate_recommendations(scored_configs)
            
            # Step 7: Generate comprehensive visualizations (dashboard, heatmap, profiling metrics)
            logger.info("Step 7: Generating comprehensive visualizations...")
            viz_output_dir = self._generate_comprehensive_visualizations(scored_configs, job_results)
            if viz_output_dir:
                logger.info(f"Comprehensive visualizations saved to: {viz_output_dir}")
            
            # Step 7b: Also call generate_results.py to ensure plots are generated with latest logic
            try:
                logger.info("Step 7b: Generating plots using generate_results.py...")
                self._call_generate_results_script()
            except Exception as e:
                logger.warning(f"Could not generate plots via generate_results.py: {e}")
                logger.warning("You can manually run: python3 scripts/generate_results.py <experiment_dir>")
            
            # Update experiment results (configurations_tested = count with full profiling results)
            best_config_mem = recommendations.get('best_configuration')
            # Extract the true mathematical best from generate_results final JSON if it exists
            final_json_path = self.work_directory / "comprehensive_results" / "results.json"
            if final_json_path.exists():
                try:
                    with open(final_json_path, "r") as f:
                        final_res = json.load(f)
                    rows = final_res
                    if isinstance(final_res, dict) and "configurations" in final_res:
                        rows = final_res["configurations"]
                    if rows and len(rows) > 0:
                        valid_rows = [
                            r for r in rows
                            if float(r.get("runtime_sec") or 0.0) > 0.0
                        ]
                        if valid_rows:
                            top_res = max(
                                valid_rows,
                                key=lambda r: float(r.get("heuristic_score") or 0.0),
                            )
                            best_config_mem = {
                                'name': top_res.get('config_name', 'Unknown'),
                                'score': top_res.get('heuristic_score', 0.0),
                                'details': {
                                    'mpi_ranks': top_res.get('mpi_ranks'),
                                    'omp_threads': top_res.get('omp_threads'),
                                    'heuristic_score_10': top_res.get('heuristic_score_10'),
                                    'runtime_sec': top_res.get('runtime_sec'),
                                }
                            }
                        else:
                            best_config_mem = None
                            logger.warning(
                                "No configuration with measured runtime > 0; skipping best-configuration summary."
                            )
                except Exception as e:
                    logger.debug(f"Could not load final results JSON for summary: {e}")

            self.experiment_results.update({
                'configurations_tested': len(job_results),
                'successful_runs': len(performance_data),
                'best_configuration': best_config_mem,
                'all_results': scored_configs,
                'comprehensive_results_dir': str(viz_output_dir) if viz_output_dir else None,
                'performance_improvement': recommendations.get('performance_improvement', 0.0),
                'speedup': recommendations.get('speedup', 1.0),
                'best_runtime': recommendations.get('best_runtime'),
                'worst_runtime': recommendations.get('worst_runtime'),
            })

            
            logger.info("🎉 Auto-tuning workflow completed successfully!")
            logger.info(f"📊 Results saved to: {self.work_directory}")
            logger.info(f"📋 Report generated: {report_path}")
            
            # Final check: Ensure plots exist, if not, try to generate them
            plot_dir = self.work_directory / "comprehensive_results"
            if not plot_dir.exists() or not any(plot_dir.glob("*.png")):
                logger.warning("⚠️  Plots not found in comprehensive_results directory")
                logger.info("Attempting to regenerate plots using generate_results.py...")
                try:
                    self._call_generate_results_script()
                except Exception as e:
                    logger.warning(f"Could not regenerate plots: {e}")
                    logger.info(f"To generate plots manually, run:")
                    logger.info(f"  python3 scripts/generate_results.py {self.work_directory} --system {self.system_name}")
            
            return {
                'experiment_id': self.experiment_id,
                'work_directory': str(self.work_directory),
                'report_path': str(report_path),
                'recommendations': recommendations,
                'summary': self.experiment_results
            }
            
        except Exception as e:
            logger.error(f"❌ Auto-tuning workflow failed: {e}")
            raise

    def _time_limit_for_profiling(self, base_time_limit: str) -> str:
        """SLURM wall time when Nsight Run 2 + LIKWID runs (optional override via --phase2-time-limit)."""
        if self.phase2_time_limit:
            return self.phase2_time_limit
        return base_time_limit

    def _resolve_profiling_arguments(self, full_arguments: str) -> str:
        """
        CLI passed to profiling batches (Phase-2, fallback full profiling, deep finalists).

        For applications that use the GPU light size cap, default matches Phase-1 prescreen scale (capped --size, iterations 2) so
        Run 1a + LIKWID + Nsight Run 2 fit GPU memory and wall time; use --phase2-args for full-scale profiling.
        """
        manual = getattr(self, "_phase2_arguments_cli", None)
        if manual is not None and str(manual).strip():
            return str(manual).strip()
        if self.application_name == "hybrid_vec" and not getattr(self, "cpu_only", True):
            out = _phase1_lightweight_arguments(
                full_arguments, phase1_size_cap=GPU_MPI_LIGHT_SIZE_CAP
            )
            if not getattr(self, "_logged_profiling_cli", False):
                logger.info(
                    "Profiling jobs: hybrid_vec default workload %r "
                    "(same cap as Phase-1). Full user scale: pass --phase2-args with your --arguments.",
                    out.strip(),
                )
                self._logged_profiling_cli = True
            return out
        return full_arguments
    
    def _generate_configurations(self, pilot_only: bool = False) -> List[Any]:
        """Generate MPI+OpenMP configurations to test"""
        logger.info("Generating configurations...")
        if pilot_only:
            logger.info("Pilot mode: Using recommended configurations only")
        else:
            logger.info("Full mode: Generating comprehensive configuration set")
        configurations = build_configurations_for_application(
            application_name=self.application_name,
            config_generator=self.config_generator,
            num_nodes=self.num_nodes,
            pilot_only=pilot_only,
        )
        logger.info(f"Generated {len(configurations)} configurations to test")
        
        # Export configurations for reference
        configs_file = self.work_directory / "generated_configurations.json"
        configs_data = []
        
        for config in configurations:
            config_data = {
                'name': config.name,
                'mpi_ranks_per_node': config.mpi_ranks_per_node,
                'omp_threads_per_rank': config.omp_threads_per_rank,
                'binding_strategy': config.binding_strategy,
                'placement_strategy': config.placement_strategy,
                'numa_policy': config.numa_policy,
                'description': config.description
            }
            if getattr(config, "num_nodes", None) is not None:
                config_data["num_nodes"] = config.num_nodes
            configs_data.append(config_data)
        
        with open(configs_file, 'w') as f:
            json.dump(configs_data, f, indent=2)
        
        logger.info(f"Exported configurations to: {configs_file}")
        return configurations
    
    def _execute_applications(self, 
                           configurations: List[Any],
                           executable_path: str,
                           arguments: str,
                           time_limit: str,
                           profiling: bool = True,
                           spread_nsight_ranks: bool = False,
                           job_name_suffix: str = "") -> Dict[str, Any]:
        """Execute application runs for all configurations using MPI.
        profiling: If False, run without Nsight (Phase-1 preliminary screening: runtime, throughput, LIKWID only).
        spread_nsight_ranks: Deep-finalist flag (job naming / future use). Multi-rank Run 2 always wraps the full mpirun with one Nsight session.
        job_name_suffix: Appended to SLURM job name (e.g. "_deep") so scripts do not overwrite prior submissions.
        """
        logger.info(f"Executing applications for {len(configurations)} configurations...")
        logger.info(f"Using MPI (Message Passing Interface) backend")
        logger.info(f"Profiling (Nsight): {'enabled' if profiling else 'disabled (Phase-1 Preliminary Screening)'}")
        if profiling:
            logger.info(f"  SLURM time limit for this batch: {time_limit}")
        if profiling and spread_nsight_ranks:
            logger.info("  Deep finalist batch: same Run 2 layout (Nsight wraps full MPI launch → profile.sqlite).")
        elif profiling:
            logger.info("  Multi-rank Run 2: Nsight wraps full mpirun (avoids rank-0-only child-process MPI hangs).")
        
        job_results = {}

        # For LULESH: compute the reference rank count (largest config) once before the loop.
        # Every config will have its -s N scaled so total zones stays constant.
        _lulesh_reference_ranks: int = 0
        if self.application_name == "lulesh":
            _lulesh_reference_ranks = reference_mpi_rank_count_from_configs(
                configurations, self.num_nodes
            )
            if _lulesh_reference_ranks > 0:
                logger.info(
                    f"LULESH problem-size auto-scaling enabled: "
                    f"reference = {_lulesh_reference_ranks} ranks "
                    f"(all configs scaled to same total zones)"
                )

        # Use specified number of nodes (or detect from environment if not specified)
        num_nodes = self.num_nodes
        if num_nodes == 1:
            # Try to detect from environment if running inside a SLURM job
            import os
            if 'SLURM_JOB_NUM_NODES' in os.environ:
                try:
                    num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
                    logger.info(f"Detected multi-node allocation from environment: {num_nodes} nodes")
                except ValueError:
                    logger.warning(f"Could not parse SLURM_JOB_NUM_NODES: {os.environ['SLURM_JOB_NUM_NODES']}")
        
        if num_nodes > 1:
            logger.info(f"🌐 Multi-node execution: {num_nodes} nodes")
            logger.info(f"   Total MPI ranks = {num_nodes} nodes × (ranks_per_node) per configuration")
        else:
            logger.info(f"🖥️  Single-node execution: 1 node")

        for i, config in enumerate(configurations):
            try:
                nodes_for_job = (
                    config.num_nodes
                    if getattr(config, "num_nodes", None) is not None
                    else num_nodes
                )
                total_mpi_ranks = nodes_for_job * config.mpi_ranks_per_node
                logger.info(f"Processing configuration {i+1}/{len(configurations)}: {config.name}")
                if nodes_for_job > 1:
                    logger.info(f"  Multi-node: {nodes_for_job} nodes × {config.mpi_ranks_per_node} ranks/node = {total_mpi_ranks} total MPI ranks")
                    logger.info(f"  OpenMP: {config.omp_threads_per_rank} threads/rank")
                    logger.info(f"  Total cores: {total_mpi_ranks * config.omp_threads_per_rank} ({nodes_for_job} nodes × {config.mpi_ranks_per_node * config.omp_threads_per_rank} cores/node)")
                else:
                    logger.info(f"  Single-node: {config.mpi_ranks_per_node} ranks × {config.omp_threads_per_rank} threads ({total_mpi_ranks} MPI ranks)")
                
                # Request GPUs if not CPU-only mode
                gpus_needed = 0 if self.cpu_only else 1
                
                # Create SLURM job configuration
                j_suffix = job_name_suffix or ""
                job_base = f"{self.application_name}_{config.name}"
                if getattr(config, "num_nodes", None) is not None:
                    job_base = f"{job_base}_n{nodes_for_job}"
                job_config = self.slurm_manager.create_job_config(
                    mpi_ranks=config.mpi_ranks_per_node,
                    omp_threads=config.omp_threads_per_rank,
                    job_name=f"{job_base}{j_suffix}",
                    time_limit=time_limit,
                    nodes=nodes_for_job,
                    account=self.account,
                    gpus_per_node=gpus_needed,
                )
                
                # If env_path is set, prepend it to PATH in job's environment variables
                if self.env_path:
                    if job_config.environment_variables is None:
                        job_config.environment_variables = {}
                    # Use $PATH to inherit existing PATH from compute node
                    job_config.environment_variables['PATH'] = f"{self.env_path}:$PATH"

                mpip_for_job, mpip_note = effective_mpip_for_job(
                    self.application_name, nodes_for_job, self.enable_mpip
                )
                if mpip_note:
                    logger.info(mpip_note)

                likwid_for_job = self.enable_likwid
                likwid_rank0_only, likwid_note = likwid_rank0_only_for_job(
                    self.application_name, nodes_for_job
                )
                if self.enable_likwid and likwid_note:
                    logger.info(likwid_note)

                job_config.additional_options = dict(job_config.additional_options or {})
                if getattr(self, "no_mpip_srun_run1a", False):
                    job_config.additional_options["mpip_mvapich_srun"] = False

                # LULESH: scale -s N per config so all configs solve the same total zones.
                effective_arguments = arguments
                if self.application_name == "lulesh" and _lulesh_reference_ranks > 0:
                    effective_arguments = scale_arguments_for_equal_total_zones(
                        arguments, total_mpi_ranks, _lulesh_reference_ranks
                    )

                # Always use full user-specified arguments (no lightweight run)
                script_path = self.slurm_manager.generate_job_script(
                    job_config, executable_path, effective_arguments, profiling=profiling, likwid_enabled=likwid_for_job, likwid_path=self.likwid_path, enable_mpip=mpip_for_job, mpip_path=self.mpip_path, light_arguments=None, multi_rank_profile=self.multi_rank_profile, nsight_duration_sec=getattr(self, 'nsight_duration_sec', None), nsight_trace_domains=getattr(self, 'nsight_trace_domains', None), spread_nsight_ranks=spread_nsight_ranks, likwid_rank0_only=likwid_rank0_only,
                )
                
                # Submit job with retry logic for QOS limits
                retry_count = 0
                max_retries = 30
                job_id = None
                
                while retry_count < max_retries:
                    try:
                        job_id = self.slurm_manager.submit_job(script_path)
                        break # Success
                    except Exception as submit_e:
                        error_str = str(submit_e)
                        if "QOSMaxSubmitJobPerUser" in error_str or "Job violates accounting/QOS policy" in error_str:
                            logger.warning(f"Hit SLURM submission limit. Waiting 60s before retrying job {config.name}...")
                            
                            # Let monitor clear out finished jobs
                            current_job_ids = [r['job_id'] for r in job_results.values() if 'job_id' in r and r.get('status') == 'SUBMITTED']
                            if current_job_ids:
                                self.slurm_manager.monitor_jobs(current_job_ids, check_interval=1)
                                
                            time.sleep(60)
                            retry_count += 1
                        else:
                            raise submit_e # Other errors, fail immediately
                            
                if job_id is None:
                    raise RuntimeError("Exceeded maximum retries for SLURM submission")
                
                # Store job information
                job_results[config.name] = {
                    'config': config,
                    'job_config': job_config,
                    'script_path': script_path,
                    'job_id': job_id,
                    'status': 'SUBMITTED',
                    'executable': executable_path
                }
                
                logger.info(f"Submitted job {job_id} for configuration {config.name}")
                
                # Small delay between submissions
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to submit job for configuration {config.name}: {e}")
                job_results[config.name] = {
                    'config': config,
                    'error': str(e),
                    'status': 'FAILED'
                }
        
        # Monitor all jobs
        logger.info("Monitoring job execution...")
        job_ids = [result['job_id'] for result in job_results.values() 
                   if 'job_id' in result]
        
        if job_ids:
            final_states = self.slurm_manager.monitor_jobs(job_ids)
            
            # Update job results with final states
            for config_name, result in job_results.items():
                if 'job_id' in result:
                    job_id = result['job_id']
                    if job_id in final_states:
                        result['status'] = final_states[job_id]
                        result['final_state'] = final_states[job_id]
                        
                        # Archive SLURM stdout/err and artifacts for terminal jobs (not only COMPLETED).
                        # FAILED/TIMEOUT runs still produce slurm-<id>.out under the experiment dir; copying
                        # into results/<job_id>/ lets generate_results and postmortems work consistently.
                        _collect_terminal = (
                            "COMPLETED",
                            "FAILED",
                            "TIMEOUT",
                            "CANCELLED",
                            "OUT_OF_MEMORY",
                            "NODE_FAIL",
                            "PREEMPTED",
                        )
                        if final_states[job_id] in _collect_terminal:
                            try:
                                results_dir = self.slurm_manager.collect_results(job_id)
                                result["results_directory"] = str(results_dir)
                                st = final_states[job_id]
                                if st == "COMPLETED":
                                    logger.info(f"Collected results for {config_name}: {results_dir}")
                                else:
                                    logger.info(
                                        "Archived artifacts for %s (job state=%s): %s — "
                                        "inspect slurm-*.err under experiment dir or sacct if metrics missing.",
                                        config_name,
                                        st,
                                        results_dir,
                                    )
                            except Exception as e:
                                logger.error(f"Failed to collect results for {config_name}: {e}")
                                result["results_collection_error"] = str(e)
        
        logger.info("Application execution completed")
        return job_results

    def _merge_phase2_job_ids_file(self, new_ids: List[str]) -> None:
        """Append job IDs to phase2_job_ids.json (dedupe, preserve order)."""
        path = self.work_directory / "phase2_job_ids.json"
        existing: List[str] = []
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    existing = [str(x) for x in data]
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
        seen = set()
        merged: List[str] = []
        for jid in existing + [str(x) for x in new_ids]:
            if jid not in seen:
                seen.add(jid)
                merged.append(jid)
        path.write_text(json.dumps(merged))
        if existing:
            logger.info(f"Updated {path}: {len(merged)} total job id(s) (added {len(new_ids)} deep id(s))")
        else:
            logger.info(f"Created {path} with {len(merged)} job id(s) (deep + any prior)")

    def _select_deep_profile_finalists(
        self,
        performance_data: Dict[str, Any],
        scored_configs: Dict[str, Any],
        job_results: Dict[str, Any],
    ) -> List[Any]:
        """Pick up to deep_profile_max_configs: fastest runtime, best heuristic, median runtime (deduped)."""
        max_n = self.deep_profile_max_configs
        eligible_names: List[str] = []
        for name, res in job_results.items():
            if res.get("status") != "COMPLETED":
                continue
            cfg = res.get("config")
            if cfg is None:
                continue
            _nn = getattr(cfg, "num_nodes", None) or self.num_nodes
            if _nn * cfg.mpi_ranks_per_node < 2:
                continue
            m = performance_data.get(name)
            if not m:
                continue
            rt = m.get("total_runtime") or 0.0
            if rt <= 0:
                continue
            eligible_names.append(name)
        if not eligible_names:
            logger.info("Deep profile: no eligible multi-rank configs with profiling + positive runtime.")
            return []
        fastest = min(eligible_names, key=lambda n: performance_data[n]["total_runtime"])
        scored_eligible = [n for n in eligible_names if n in scored_configs]
        if scored_eligible:
            best_h = max(
                scored_eligible,
                key=lambda n: float(scored_configs[n].get("score", float("-inf"))),
            )
        else:
            best_h = fastest
        by_rt = sorted(eligible_names, key=lambda n: performance_data[n]["total_runtime"])
        median_n = by_rt[len(by_rt) // 2]
        picks: List[str] = []
        seen = set()
        for cand in (fastest, best_h, median_n):
            if cand in seen:
                continue
            seen.add(cand)
            picks.append(cand)
            if len(picks) >= max_n:
                break
        cfg_by_name = {n: job_results[n]["config"] for n in picks}
        return [cfg_by_name[n] for n in picks]

    def _maybe_run_deep_profile_finalists(
        self,
        job_results: Dict[str, Any],
        executable_path: str,
        arguments: str,
        time_limit: str,
    ) -> Dict[str, Any]:
        if not self.deep_profile_finalists:
            return job_results
        if self.no_nsight:
            logger.info("Deep profile finalists skipped (--no-nsight).")
            return job_results
        if not self.multi_rank_profile:
            logger.info("Deep profile finalists skipped (multi-rank profiling disabled).")
            return job_results
        logger.info("Deep profile phase: coarse metrics from main batch to select finalists...")
        coarse_perf = self._extract_performance_metrics(job_results)
        self._maybe_use_cpu_weight_scoring_when_gpu_absent(coarse_perf)
        coarse_scored = self._score_configurations(coarse_perf)
        finalists = self._select_deep_profile_finalists(coarse_perf, coarse_scored, job_results)
        if not finalists:
            return job_results
        names = [c.name for c in finalists]
        logger.info(
            "Deep profile phase: submitting spread Nsight jobs for %d finalist(s): %s",
            len(finalists),
            names,
        )
        deep_tl = self.deep_profile_time_limit or self._time_limit_for_profiling(time_limit)
        deep_results = self._execute_applications(
            finalists,
            executable_path,
            self._resolve_profiling_arguments(arguments),
            deep_tl,
            profiling=True,
            spread_nsight_ranks=True,
            job_name_suffix="_deep",
        )
        deep_ids = [str(r["job_id"]) for r in deep_results.values() if "job_id" in r]
        deep_path = self.work_directory / "deep_profile_job_ids.json"
        deep_path.write_text(json.dumps(deep_ids))
        logger.info(f"Deep job IDs written to {deep_path}")
        phase2_path = self.work_directory / "phase2_job_ids.json"
        if not phase2_path.exists():
            bootstrap = [str(r["job_id"]) for r in job_results.values() if "job_id" in r]
            phase2_path.write_text(json.dumps(bootstrap))
            logger.warning(
                "phase2_job_ids.json was missing; bootstrapped from current job_results before merging deep IDs."
            )
        self._merge_phase2_job_ids_file(deep_ids)
        summary = {"finalists": names, "job_ids": deep_ids, "time_limit": deep_tl}
        (self.work_directory / "deep_profile_summary.json").write_text(json.dumps(summary, indent=2))
        for cfg in finalists:
            n = cfg.name
            dr = deep_results.get(n)
            if not dr:
                continue
            if dr.get("status") == "COMPLETED" and dr.get("results_directory"):
                job_results[n] = dr
                logger.info("Deep profile: using job %s for config %s", dr.get("job_id"), n)
            else:
                logger.warning(
                    "Deep profile job for %s did not complete successfully (status=%s); keeping main-batch results.",
                    n,
                    dr.get("status", "unknown"),
                )
        return job_results
    
    def _extract_performance_metrics(self, job_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from completed application runs"""
        logger.info("Extracting performance metrics...")
        
        performance_data = {}
        
        for config_name, result in job_results.items():
            if result.get('status') == 'COMPLETED' and 'results_directory' in result:
                try:
                    results_dir = Path(result['results_directory'])
                    # Also store work_directory for fallback file search
                    work_directory = self.work_directory
                    
                    # Look for profiling data in multiple locations
                    profiling_files = list(results_dir.rglob("*.sqlite"))
                    
                    # Get job_id from the result
                    job_id = result.get('job_id', results_dir.name)
                    
                    # Check experiment's profiling directory (NEW: primary location)
                    experiment_profiling_dir = work_directory / "profiling" / str(job_id)
                    if experiment_profiling_dir.exists():
                        profiling_files.extend(list(experiment_profiling_dir.glob("*.sqlite")))
                    
                    # Also check the global profiling directory (legacy location)
                    global_profiling_dir = work_directory.parent / "profiling" / str(job_id)
                    if global_profiling_dir.exists():
                        profiling_files.extend(list(global_profiling_dir.glob("*.sqlite")))
                    
                    # Also check relative profiling path (another legacy location)
                    relative_profiling_dir = Path("profiling") / str(job_id)
                    if relative_profiling_dir.exists():
                        profiling_files.extend(list(relative_profiling_dir.glob("*.sqlite")))
                    
                    if profiling_files:
                        # Multi-rank: one profile.sqlite per job (full-job Nsight session)
                        logger.info(f"Found {len(profiling_files)} profiling file(s) for {config_name}")
                        
                        # Aggregate metrics from profiled rank(s)
                        all_metrics = []
                        for pf in profiling_files:
                            try:
                                logger.info(f"Extracting metrics from {pf}")
                                with NsightMetricsExtractor(pf) as extractor:
                                    metrics = extractor.extract_all_metrics(backend_prefix="MPI_")
                                    all_metrics.append(metrics)
                            except Exception as e:
                                logger.warning(f"Failed to extract from {pf}: {e}")
                        
                        if all_metrics:
                            # Use first profile as base, aggregate others
                            primary_metrics = all_metrics[0]
                            
                            # Multiple sqlite files (e.g. legacy runs): aggregate if >1
                            if len(all_metrics) > 1:
                                logger.info(f"Aggregating metrics from {len(all_metrics)} rank profile(s)")
                                total_cpu_util = sum(m.cpu_utilization for m in all_metrics)
                                avg_cpu_util = total_cpu_util / len(all_metrics)
                                
                                total_runtime = max(m.total_runtime for m in all_metrics)
                                # Use max (bottleneck rank), not sum: comm time cannot exceed runtime
                                total_comm_time = max(m.mpi_comm_time for m in all_metrics)
                                stalls = [m.thread_stall_time for m in all_metrics]
                                total_stall_time = sum(stalls) / len(stalls) if stalls else 0.0
                            else:
                                avg_cpu_util = primary_metrics.cpu_utilization
                                total_runtime = primary_metrics.total_runtime
                                total_comm_time = primary_metrics.mpi_comm_time
                                total_stall_time = primary_metrics.thread_stall_time
                            
                            # Use best available numa_efficiency (LIKWID may be found only when profile path is experiment_dir/profiling/job_id)
                            best_numa = max((m.numa_efficiency for m in all_metrics), default=primary_metrics.numa_efficiency)
                            # Use best GPU/OpenMP across ranks when aggregating (e.g. max gpu_utilization)
                            best_gpu_util = max((m.gpu_utilization for m in all_metrics), default=primary_metrics.gpu_utilization)
                            best_gpu_bw = max((m.gpu_memory_bandwidth for m in all_metrics), default=primary_metrics.gpu_memory_bandwidth)
                            best_openmp_eff = max((m.openmp_work_efficiency for m in all_metrics), default=primary_metrics.openmp_work_efficiency)
                            openmp_instrumented = any(
                                getattr(m, "openmp_instrumented", False) for m in all_metrics
                            )
                            best_gpu_busy = max(
                                (float(getattr(m, "gpu_busy_time", 0.0) or 0.0) for m in all_metrics),
                                default=float(getattr(primary_metrics, "gpu_busy_time", 0.0) or 0.0),
                            )
                            best_cupti_procs = max(
                                (int(getattr(m, "cupti_process_count", 0) or 0) for m in all_metrics),
                                default=0,
                            )
                            best_gpu_span = max(
                                (float(getattr(m, "gpu_active_span", 0.0) or 0.0) for m in all_metrics),
                                default=0.0,
                            )
                            # Convert to dictionary format (including distribution analysis and GPU/OpenMP for δ, ε)
                            best_mem_bw = max(
                                (float(getattr(m, "memory_bandwidth", 0.0) or 0.0) for m in all_metrics),
                                default=float(getattr(primary_metrics, "memory_bandwidth", 0.0) or 0.0),
                            )
                            best_mpip_app = max(
                                (float(getattr(m, "mpip_app_time", 0.0) or 0.0) for m in all_metrics),
                                default=float(getattr(primary_metrics, "mpip_app_time", 0.0) or 0.0),
                            )
                            best_mpip_frac = max(
                                (
                                    float(getattr(m, "mpip_max_mpi_wall_fraction", 0.0) or 0.0)
                                    for m in all_metrics
                                ),
                                default=float(
                                    getattr(primary_metrics, "mpip_max_mpi_wall_fraction", 0.0) or 0.0
                                ),
                            )
                            metrics_dict = {
                                'mpi_comm_time': total_comm_time,
                                'mpip_app_time': best_mpip_app,
                                'mpip_max_mpi_wall_fraction': best_mpip_frac,
                                'total_runtime': total_runtime,
                                'profiling_trace_present': True,
                                'cpu_utilization': avg_cpu_util,
                                'memory_local_accesses': primary_metrics.memory_local_accesses,
                                'memory_total_accesses': primary_metrics.memory_total_accesses,
                                'numa_efficiency': best_numa,
                                'memory_bandwidth': best_mem_bw,
                                'memory_read_bandwidth': max(
                                    (float(getattr(m, "memory_read_bandwidth", 0.0) or 0.0) for m in all_metrics),
                                    default=float(getattr(primary_metrics, "memory_read_bandwidth", 0.0) or 0.0),
                                ),
                                'memory_write_bandwidth': max(
                                    (float(getattr(m, "memory_write_bandwidth", 0.0) or 0.0) for m in all_metrics),
                                    default=float(getattr(primary_metrics, "memory_write_bandwidth", 0.0) or 0.0),
                                ),
                                'thread_stall_time': total_stall_time,
                                'load_imbalance': primary_metrics.load_imbalance,
                                'cache_miss_rate': primary_metrics.cache_miss_rate,
                                # GPU metrics (δ) from CUPTI when trace includes cuda,cublas
                                'gpu_utilization': best_gpu_util,
                                'gpu_busy_time': best_gpu_busy,
                                'gpu_memory_bandwidth': best_gpu_bw,
                                'cupti_process_count': best_cupti_procs,
                                'gpu_active_span': best_gpu_span,
                                # OpenMP metrics (ε) from Nsight trace-derived efficiency
                                'openmp_work_efficiency': best_openmp_eff,
                                'openmp_instrumented': openmp_instrumented,
                                'openmp_work_time': primary_metrics.openmp_work_time,
                                # Distribution analysis data for bottleneck detection
                                'distribution_stats': primary_metrics.distribution_stats,
                                'per_rank_breakdown': primary_metrics.per_rank_breakdown,
                                'per_thread_breakdown': primary_metrics.per_thread_breakdown,
                                'per_stream_breakdown': primary_metrics.per_stream_breakdown,
                                'outliers': primary_metrics.outliers,
                                'bottleneck_details': primary_metrics.bottleneck_details,
                                # Number of profiled ranks (1-3 for multi-rank: ranks 0, N/2, N-1)
                                'num_rank_profiles': len(all_metrics)
                            }
                            if best_mpip_frac <= 0.001 and self.work_directory:
                                try:
                                    mpip_extra = NsightMetricsExtractor.load_mpip_comm_metrics_from_experiment(
                                        Path(self.work_directory).resolve(),
                                        str(job_id),
                                    )
                                    frac = float(
                                        mpip_extra.get("mpip_max_mpi_wall_fraction", 0.0) or 0.0
                                    )
                                    if frac > 0.001:
                                        metrics_dict["mpip_max_mpi_wall_fraction"] = frac
                                        metrics_dict["mpip_app_time"] = float(
                                            mpip_extra.get("max_rank_app_time", 0.0) or 0.0
                                        )
                                        tc = float(mpip_extra.get("total_comm_time", 0.0) or 0.0)
                                        if tc > 0:
                                            metrics_dict["mpi_comm_time"] = tc
                                except Exception:
                                    pass
                            try:
                                from autotuner.core.profiling_refinement import (
                                    peak_nvidia_smi_util_from_profile_dir,
                                    refine_gpu_utilization,
                                )
                            except ImportError:
                                peak_nvidia_smi_util_from_profile_dir = None
                                refine_gpu_utilization = None
                            prof_dir = (
                                Path(self.work_directory) / "profiling" / str(job_id)
                                if self.work_directory
                                else results_dir / "profiling"
                            )
                            if refine_gpu_utilization and prof_dir.exists():
                                peak_smi = (
                                    peak_nvidia_smi_util_from_profile_dir(prof_dir)
                                    if peak_nvidia_smi_util_from_profile_dir
                                    else 0.0
                                )
                                jc = self.configurations.get(config_name)
                                mpi_r = (
                                    int(getattr(jc, "mpi_ranks_per_node", 1) or 1)
                                    * int(getattr(jc, "num_nodes", 1) or 1)
                                    if jc
                                    else 1
                                )
                                app_rt = float(
                                    metrics_dict.get("total_runtime")
                                    or result.get("runtime", 0.0)
                                    or 0.0
                                )
                                if app_rt <= 0:
                                    basic = self._extract_basic_metrics_from_output(
                                        results_dir, config_name, work_directory=self.work_directory
                                    )
                                    if basic:
                                        app_rt = float(basic.get("total_runtime", 0.0) or 0.0)
                                gu, gu_src = refine_gpu_utilization(
                                    best_gpu_busy,
                                    application_runtime=app_rt,
                                    trace_runtime=float(metrics_dict.get("total_runtime", 0.0) or 0.0),
                                    mpi_comm_time=float(metrics_dict.get("mpi_comm_time", 0.0) or 0.0),
                                    mpip_wall_fraction=float(
                                        metrics_dict.get("mpip_max_mpi_wall_fraction", 0.0) or 0.0
                                    ),
                                    total_mpi_ranks=max(1, mpi_r),
                                    cupti_process_count=best_cupti_procs,
                                    gpu_active_span=best_gpu_span,
                                    nvidia_smi_util=peak_smi,
                                    cpu_utilization=float(
                                        metrics_dict.get("cpu_utilization", 0.0) or 0.0
                                    ),
                                    profiling_phase_aligned=mpi_r <= 1,
                                )
                                if gu > 0:
                                    metrics_dict["gpu_utilization"] = gu
                                    metrics_dict["gpu_utilization_source"] = gu_src
                            if best_mem_bw <= 0.1 and self.work_directory:
                                try:
                                    from autotuner.core.likwid_profiler import resolve_memory_bandwidth_for_job
                                    bw, _src = resolve_memory_bandwidth_for_job(
                                        Path(self.work_directory), str(job_id), best_mem_bw
                                    )
                                    if bw > best_mem_bw:
                                        metrics_dict["memory_bandwidth"] = bw
                                except Exception:
                                    pass

                            performance_data[config_name] = metrics_dict
                            
                            # Export detailed metrics report for first profile
                            metrics_dir = results_dir / "extracted_metrics"
                            with NsightMetricsExtractor(profiling_files[0]) as export_extractor:
                                export_extractor.export_metrics_report(metrics_dir, primary_metrics)
                        else:
                            logger.warning(f"Failed to extract metrics from any profile files for {config_name}")
                            # Fallback: Extract basic metrics from application output
                            basic_metrics = self._extract_basic_metrics_from_output(results_dir, config_name, work_directory=self.work_directory)
                            if basic_metrics:
                                performance_data[config_name] = basic_metrics
                                logger.info(f"Extracted basic metrics from output for {config_name}: runtime={basic_metrics.get('total_runtime', 0):.3f}s")
                            
                    else:
                        logger.warning(f"⚠️  No Nsight SQLite (.sqlite) found for {config_name} in {results_dir}")
                        logger.warning(
                            "    Common causes: `nsys` not on PATH on compute nodes (add Nsight bin to --env-path); "
                            "Run 2 skipped when NSYS_EXE is empty; trace failed or runtime very short."
                        )
                        logger.warning(
                            "    Fallback: basic metrics from application stdout (miniMD PERF_SUMMARY, LULESH, HYBRID_VEC_GPU lines) "
                            "when available."
                        )
                        # Fallback: Extract basic metrics from application output
                        if config_name not in performance_data:
                            basic_metrics = self._extract_basic_metrics_from_output(results_dir, config_name, work_directory=self.work_directory)
                            if basic_metrics:
                                performance_data[config_name] = basic_metrics
                                logger.info(f"Extracted basic metrics from output for {config_name}: runtime={basic_metrics.get('total_runtime', 0):.3f}s")
                            else:
                                logger.warning(f"Could not extract basic metrics from output for {config_name}")

                        # Validate runtime to prevent false positives from crashed runs
                        if config_name in performance_data:
                            rt = performance_data[config_name].get('total_runtime', 0.0)
                            if rt > 0 and rt < 0.1:
                                logger.error(f"❌ REJECTING {config_name}: Runtime {rt:.2f}s is suspiciously fast (likely crashed/aborted).")
                                del performance_data[config_name]
                                continue
                        
                except Exception as e:
                    logger.error(f"Failed to extract metrics for {config_name}: {e}")
                    continue

        
        logger.info(f"Extracted metrics for {len(performance_data)} configurations")
        return performance_data
    
    def _extract_basic_metrics_from_output(self, results_dir: Path, config_name: str, work_directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Extract basic metrics from application stdout when profiling data is unavailable.
        Prefers last SPARSE/HYBRID_VEC_GPU line; ignores Time=0 when step was CANCELLED/Killed.
        Falls back to run1_<job_id>.out when present."""
        job_id = results_dir.name
        # Prefer SLURM output in this job's results dir, then work directory (match by job_id)
        slurm_file = results_dir / f"slurm-{job_id}.out"
        if not slurm_file.exists() and work_directory:
            slurm_file = work_directory / f"slurm-{job_id}.out"
        if not slurm_file.exists():
            work_dir = results_dir.parent.parent
            if work_dir.exists():
                slurm_file = work_dir / f"slurm-{job_id}.out"
        if not slurm_file.exists():
            logger.debug(f"No output files found for {config_name} in {results_dir}")
            return None

        def _read_rt_tp(path: Path) -> tuple[float, float]:
            try:
                return parse_job_stdout_runtime_throughput(path.read_text())
            except Exception:
                return 0.0, 0.0

        try:
            runtime, throughput = _read_rt_tp(slurm_file)
            if runtime == 0.0:
                run1_path = results_dir / f"run1_{job_id}.out"
                if not run1_path.exists() and work_directory:
                    run1_path = work_directory / f"run1_{job_id}.out"
                if run1_path.exists():
                    runtime, throughput = _read_rt_tp(run1_path)
            output = slurm_file.read_text()
        except Exception as e:
            logger.debug(f"Could not read output file: {e}")
            return None

        metrics: Dict[str, Any] = {}
        if runtime > 0:
            metrics["total_runtime"] = runtime
        if throughput > 0:
            metrics["throughput_gflops"] = throughput
        metrics.update(infer_stdout_metadata(output))
        
        # If we got at least runtime, create basic metrics structure
        if 'total_runtime' in metrics:
            # LIKWID is independent of Nsight; use for locality when no profiling (e.g. multi-rank)
            exp_dir = work_directory if work_directory else (results_dir.parent.parent if results_dir.parent else None)
            likwid_numa = 0.0
            if exp_dir and exp_dir.exists():
                try:
                    from autotuner.core.likwid_profiler import get_numa_efficiency_for_job
                    likwid_numa = get_numa_efficiency_for_job(exp_dir, job_id)
                except Exception:
                    pass
            rt = float(metrics["total_runtime"])
            mpi_comm_time = 0.0
            if exp_dir and exp_dir.exists():
                try:
                    mpip = NsightMetricsExtractor.load_mpip_comm_metrics_from_experiment(
                        exp_dir.resolve(), str(job_id)
                    )
                    tc = float(mpip.get("total_comm_time", 0.0) or 0.0)
                    if tc > 0.0 and rt > 0.0:
                        mpi_comm_time = min(tc, rt)
                        if tc > rt:
                            logger.debug(
                                f"mpiP MPI time {tc:.4f}s capped to wall runtime {rt:.4f}s "
                                f"(job {job_id})"
                            )
                except Exception as e:
                    logger.debug(f"mpiP merge for basic metrics failed for job {job_id}: {e}")
            # Create minimal metrics dict compatible with scoring system
            return {
                'total_runtime': metrics['total_runtime'],
                'mpi_comm_time': mpi_comm_time,  # from mpiP when Nsight Run 2 skipped
                'cpu_utilization': 0.0,  # unknown without Nsight (β uses neutral via profiling_trace_present)
                'memory_local_accesses': 0,  # Unknown
                'memory_total_accesses': 0,  # Unknown
                'numa_efficiency': likwid_numa,  # From LIKWID when available (independent of Nsight)
                'thread_stall_time': 0.0,  # Unknown
                'load_imbalance': 0.0,  # Unknown
                'cache_miss_rate': 0.0,  # Unknown
                'gpu_utilization': 0.0,  # No GPU for this application
                'gpu_memory_bandwidth': 0.0,
                'openmp_work_efficiency': 0.0,  # unknown without Nsight (ε uses neutral when trace missing)
                'profiling_trace_present': False,
                'throughput_gflops': metrics.get('throughput_gflops', 0.0),
                'execution_time': metrics['total_runtime'],
                'application_type': metrics.get('application_type', 'UNKNOWN'),
                'problem_size': metrics.get('problem_size', 0),
                'iterations': metrics.get('iterations', 0),
                # Distribution analysis (empty since we don't have detailed data)
                'distribution_stats': {},
                'per_rank_breakdown': {},
                'per_thread_breakdown': {},
                'per_stream_breakdown': {},
                'outliers': {},
                'bottleneck_details': {}
            }
        
        return None

    def _log_phase1_failure_diagnostics(
        self,
        job_results: Dict[str, Any],
        system_config: Optional[Dict[str, Any]],
    ) -> None:
        """Print SLURM stderr/stdout excerpt and paths checked when Phase-1 aborts."""
        sample_jid: Optional[str] = None
        for data in job_results.values():
            if isinstance(data, dict) and data.get("job_id") is not None:
                sample_jid = str(data["job_id"])
                break
        if not sample_jid:
            return
        log_path, excerpt = read_slurm_log_excerpt(
            sample_jid,
            system_config=system_config,
            experiment_dir=self.work_directory,
        )
        if log_path and excerpt:
            logger.error(
                "Phase-1 diagnostic — SLURM log for job %s (%s), last lines:\n%s",
                sample_jid,
                log_path,
                excerpt,
            )
        else:
            checked = slurm_log_candidate_paths(
                sample_jid,
                system_config=system_config,
                experiment_dir=self.work_directory,
            )
            logger.error(
                "Phase-1 diagnostic — no SLURM log read for job %s. Checked: %s",
                sample_jid,
                ", ".join(str(p) for p in checked[:12]),
            )
        try:
            import subprocess as _sp

            _np = _sp.run(
                [
                    "sacct",
                    "-j",
                    str(sample_jid),
                    "-n",
                    "--parsable2",
                    "--format=JobID,NodeList,ExitCode",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if _np.returncode == 0 and _np.stdout.strip():
                logger.error(
                    "Phase-1 diagnostic — sacct NodeList for job %s: %s",
                    sample_jid,
                    _np.stdout.strip().replace("\n", " | "),
                )
        except Exception:
            pass

    def _all_job_logs_show_binary_exec_failure(
        self, job_results: Dict[str, Any], max_chars: int = 200_000
    ) -> bool:
        """
        True when every job that has readable SLURM/run logs shows ``Exec format error`` or
        ``cannot execute binary file`` for the application — i.e. the binary never started on
        compute nodes (wrong architecture / bad ELF). Used to fail fast instead of burning
        Phase-2 GPU wall time when Phase-1 already had no timings.
        """
        markers = (
            "Exec format error",
            "cannot execute binary file",
            "cannot execute the binary file",
        )
        root = Path(self.work_directory)
        readable = 0
        for data in job_results.values():
            if not isinstance(data, dict):
                continue
            jid = data.get("job_id")
            if jid is None:
                continue
            jid = str(jid)
            chunks: List[str] = []
            paths: List[Path] = []
            res_dir = data.get("results_directory")
            if res_dir:
                rp = Path(res_dir)
                paths.extend(
                    [
                        rp / f"run1_{jid}.out",
                        rp / f"slurm-{jid}.out",
                        rp / f"slurm-{jid}.err",
                    ]
                )
            paths.extend(
                [
                    root / f"run1_{jid}.out",
                    root / f"slurm-{jid}.out",
                    root / f"slurm-{jid}.err",
                    root / "results" / jid / f"run1_{jid}.out",
                    root / "results" / jid / f"slurm-{jid}.out",
                    root / "results" / jid / f"slurm-{jid}.err",
                ]
            )
            seen: Set[str] = set()
            for p in paths:
                try:
                    key = str(p.resolve())
                except OSError:
                    key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                if not p.is_file():
                    continue
                try:
                    chunks.append(p.read_text(encoding="utf-8", errors="replace")[:max_chars])
                except OSError:
                    continue
            text = "\n".join(chunks)
            if not text.strip():
                continue
            readable += 1
            if not any(m in text for m in markers):
                return False
        return readable > 0

    def _maybe_use_cpu_weight_scoring_when_gpu_absent(
        self, performance_data: Dict[str, Any], gpu_util_threshold: float = 0.02
    ) -> None:
        """
        If runs are not ``--cpu-only`` but every Nsight-derived row shows negligible GPU utilization,
        the δ term drags the composite down (e.g. Exec format error: 0 CUDA kernels). In that case
        rebuild the scorer with ``cpu_only=True`` so δ and ε are redistributed to α and β — matching
        the numeric balance of CPU MPI+OpenMP apps (e.g. sparse_matrix) until a real GPU workload exists.
        """
        if self.cpu_only or not performance_data:
            return
        max_gpu = 0.0
        for m in performance_data.values():
            if not isinstance(m, dict):
                continue
            try:
                max_gpu = max(max_gpu, float(m.get("gpu_utilization") or 0.0))
            except (TypeError, ValueError):
                continue
        if max_gpu > gpu_util_threshold:
            return
        logger.warning(
            "Negligible GPU utilization in all traces (max=%.4f). Using CPU-only heuristic weights "
            "(δ and ε redistributed to α and β) so scores align with MPI+OpenMP–dominated apps. "
            "Rebuild/run a working GPU binary on compute nodes to restore full δ GPU scoring.",
            max_gpu,
        )
        self._gpu_scoring_fallback = True
        try:
            self.experiment_results["gpu_scoring_cpu_weight_fallback"] = True
        except Exception:
            pass
        lk = bool(getattr(self, "enable_likwid", False))
        if self.adaptive_weights:
            self.scorer = AdaptiveConfigurationScorer(
                enable_adaptation=True, enable_locality=lk, cpu_only=True
            )
        else:
            self.scorer = ConfigurationScorer(enable_locality=lk, cpu_only=True)

    def _score_configurations(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score configurations using the heuristic scoring function"""
        logger.info("Scoring configurations using heuristic function...")
        
        scored_configs = {}

        # γ locality: bandwidth-based score needs max LIKWID BW across configs as reference
        if getattr(self.scorer, "enable_locality", False) and performance_data:
            max_bw = max(
                (float(m.get("memory_bandwidth") or 0.0) for m in performance_data.values()),
                default=0.0,
            )
            if max_bw > 0 and hasattr(self.scorer, "set_reference_bandwidth"):
                self.scorer.set_reference_bandwidth(max_bw)
        
        for config_name, metrics in performance_data.items():
            try:
                # Get configuration details from job results
                config_obj = None
                for result in self.experiment_results.get('all_results', {}).values():
                    # The stored config object is likely an MPIOpenMPConfig dataclass instance, not a dict
                    config_entry = result.get('config')
                    if hasattr(config_entry, 'name') and config_entry.name == config_name:
                        config_obj = config_entry
                        break
                
                if config_obj:
                    mpi_ranks = config_obj.mpi_ranks_per_node
                    omp_threads = config_obj.omp_threads_per_rank
                    
                    # Score the configuration
                    score = self.scorer.score_configuration(
                        config_name, mpi_ranks, omp_threads, metrics
                    )
                    
                    scored_configs[config_name] = {
                        'config': config_obj,
                        'metrics': metrics,  # Includes distribution analysis data
                        'score': score,
                        'mpi_ranks': mpi_ranks,
                        'omp_threads': omp_threads,
                        'performance_metrics': metrics  # For report generator compatibility
                    }
                    
                    logger.info(f"Scored {config_name}: {score:.4f}")
                else:
                    logger.warning(f"Could not find configuration object for {config_name}")
                    
            except Exception as e:
                logger.error(f"Failed to score configuration {config_name}: {e}")
                continue
        
        logger.info(f"Scored {len(scored_configs)} configurations")
        return scored_configs
    
    def _generate_final_report(self, scored_configs: Dict[str, Any]) -> Path:
        """Generate the final comprehensive report"""
        logger.info("Generating final report...")
        
        # Prepare data for report generation
        report_data = self.experiment_results.copy()
        report_data['all_results'] = scored_configs
        
        # Get best configuration
        if scored_configs:
            best_result = self.scorer.get_best_configuration()
            if best_result:
                best_config_name, best_score = best_result
                details = scored_configs.get(best_config_name, {}) or {}
                try:
                    _rt = float(
                        details.get("runtime_sec")
                        or details.get("total_runtime")
                        or 0.0
                    )
                except (TypeError, ValueError):
                    _rt = 0.0
                if _rt > 0.0:
                    best_config_entry = {
                        'name': best_config_name,
                        'score': best_score,
                        'details': {**details, 'runtime_sec': _rt},
                    }
                    report_data['best_configuration'] = best_config_entry
                    self.experiment_results['best_configuration'] = best_config_entry
                else:
                    report_data['best_configuration'] = None
                    self.experiment_results['best_configuration'] = None
        
        # Generate report (pass scorer so Comm/Thread/Locality scores match component_scores)
        report_path = self.report_generator.generate_report(
            report_data, self.work_directory, scorer=self.scorer
        )
        
        # Export scoring results
        scoring_dir = self.work_directory / "scoring_results"
        self.scorer.export_results(scoring_dir)
        
        logger.info(f"Generated final report: {report_path}")
        return Path(report_path)
    
    def _generate_recommendations(self, scored_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        logger.info("Generating optimization recommendations...")
        
        recommendations = {
            'best_configuration': None,
            'performance_improvement': 0.0,
            'configuration_ranking': [],
            'optimization_suggestions': []
        }
        
        if not scored_configs:
            return recommendations
        
        # Get best configuration
        best_config_name = None
        best_result = self.scorer.get_best_configuration()
        if best_result:
            best_config_name, best_score = best_result
            best_config = scored_configs[best_config_name]
            recommendations['best_configuration'] = {
                'name': best_config_name,
                'score': best_score,
                'mpi_ranks': best_config['mpi_ranks'],
                'omp_threads': best_config['omp_threads']
            }
            
            # Calculate performance improvement based on runtime (not score)
            runtimes = []
            for config_name, config_data in scored_configs.items():
                metrics = config_data.get('metrics') or config_data.get('performance_metrics') or {}
                rt = float(
                    metrics.get('total_runtime')
                    or metrics.get('execution_time')
                    or config_data.get('total_runtime')
                    or 0.0
                )
                if rt > 0:
                    runtimes.append((config_name, rt))
            
            if runtimes:
                # Find best (fastest) and worst (slowest) runtime
                best_runtime = min(r[1] for r in runtimes)
                worst_runtime = max(r[1] for r in runtimes)
                if best_runtime > 0:
                    # Speedup = worst / best - 1 (percentage faster than worst)
                    improvement = ((worst_runtime - best_runtime) / best_runtime) * 100
                    recommendations['performance_improvement'] = improvement
                    recommendations['speedup'] = worst_runtime / best_runtime
                    recommendations['best_runtime'] = best_runtime
                    recommendations['worst_runtime'] = worst_runtime

        
        # Get configuration ranking
        ranking = self.scorer.get_configuration_ranking()
        recommendations['configuration_ranking'] = [
            {
                'rank': i + 1,
                'name': name,
                'score': score,
                'mpi_ranks': data['mpi_ranks'],
                'omp_threads': data['omp_threads']
            }
            for i, (name, score, data) in enumerate(ranking)
        ]
        
        # Generate optimization suggestions
        if best_config_name:
            analysis = self.scorer.get_detailed_analysis(best_config_name)
            if analysis:
                recommendations['optimization_suggestions'] = analysis.get('recommendations', [])
        
        logger.info(f"Generated recommendations for {len(scored_configs)} configurations")
        return recommendations
    
    def _generate_comprehensive_visualizations(self, 
                                                scored_configs: Dict[str, Any],
                                                job_results: Dict[str, Any]) -> Optional[Path]:
        """
        Generate comprehensive visualizations including dashboard, heatmap, and profiling metrics.
        Integrates functionality from generate_results.py for end-to-end analysis.
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization packages not available. Install pandas, matplotlib, seaborn.")
            return None
        
        logger.info("Generating comprehensive visualizations...")
        
        try:
            output_dir = self.work_directory / "comprehensive_results"
            output_dir.mkdir(exist_ok=True)
            
            # Collect metrics from scored_configs
            all_metrics = []
            
            for config_name, config_data in scored_configs.items():
                metrics = config_data.get('metrics', {})
                mpi_ranks = config_data.get('mpi_ranks', 1)
                omp_threads = config_data.get('omp_threads', 1)
                
                # Get job_id
                job_data = job_results.get(config_name, {})
                job_id = None
                if 'results_directory' in job_data:
                    job_id = Path(job_data['results_directory']).name
                
                # Initialize metrics
                runtime = metrics.get('execution_time', metrics.get('total_runtime', 0.0))
                throughput = metrics.get('throughput_gflops', 0.0)
                osrt_events = 0
                sched_events = 0
                sched_in = 0
                sched_out = 0
                unique_cpus = 0
                num_profiles = 0
                
                if job_id and (runtime <= 0 or throughput <= 0):
                    slurm_out = find_slurm_log_for_job(
                        self.work_directory, job_id,
                        system_config=getattr(self.slurm_manager, "system_config", None),
                    )
                    if slurm_out and slurm_out.is_file():
                        try:
                            rt, tp = parse_slurm_file_runtime_throughput(slurm_out)
                            if runtime <= 0 and rt > 0:
                                runtime = rt
                            if throughput <= 0 and tp > 0:
                                throughput = tp
                        except Exception as e:
                            logger.debug(f"Could not parse SLURM output: {e}")
                
                # Extract profiling data from SQLite files
                if job_id:
                    for profiling_dir in [self.work_directory / "profiling" / str(job_id),  # Primary: experiment dir
                                          self.work_directory.parent / "profiling" / str(job_id),  # Legacy
                                          Path("profiling") / str(job_id)]:  # Legacy relative
                        if profiling_dir and profiling_dir.exists():
                            sqlite_files = list(profiling_dir.glob("*.sqlite"))
                            num_profiles = len(sqlite_files)
                            
                            for sqlite_file in sqlite_files:
                                try:
                                    conn = sqlite3.connect(str(sqlite_file))
                                    cursor = conn.cursor()
                                    
                                    # Get OSRT_API count
                                    try:
                                        cursor.execute("SELECT COUNT(*) FROM OSRT_API")
                                        osrt_events += cursor.fetchone()[0]
                                    except:
                                        pass
                                    
                                    # Get SCHED_EVENTS data
                                    try:
                                        cursor.execute("SELECT COUNT(*) FROM SCHED_EVENTS")
                                        sched_events += cursor.fetchone()[0]
                                        cursor.execute("SELECT SUM(isSchedIn), SUM(isSchedOut) FROM SCHED_EVENTS")
                                        row = cursor.fetchone()
                                        if row:
                                            sched_in += row[0] or 0
                                            sched_out += row[1] or 0
                                        cursor.execute("SELECT COUNT(DISTINCT cpu) FROM SCHED_EVENTS")
                                        cpus = cursor.fetchone()[0]
                                        if cpus > unique_cpus:
                                            unique_cpus = cpus
                                    except:
                                        pass
                                    
                                    conn.close()
                                except Exception as e:
                                    logger.debug(f"Could not read {sqlite_file}: {e}")
                            break
                
                config_metrics = {
                    'config_name': config_name,
                    'mpi_ranks': mpi_ranks,
                    'omp_threads': omp_threads,
                    'total_cores': mpi_ranks * omp_threads,
                    'runtime_sec': runtime,
                    'throughput_gflops': throughput,
                    'osrt_events': osrt_events,
                    'sched_events': sched_events,
                    'sched_in': sched_in,
                    'sched_out': sched_out,
                    'unique_cpus': unique_cpus,
                    'num_rank_profiles': num_profiles,
                    'job_id': job_id,
                }
                
                all_metrics.append(config_metrics)
            
            if not all_metrics:
                logger.warning("No metrics available for visualization")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(all_metrics)
            
            # Extract ACTUAL heuristic component scores from scorer
            # These are the real α, β, γ, δ, ε values used in scoring
            alpha_scores = []
            beta_scores = []
            gamma_scores = []
            delta_scores = []
            epsilon_scores = []
            actual_heuristic_scores = []
            
            for config_name in df['config_name']:
                if config_name in self.scorer.scored_configurations:
                    scores = self.scorer.scored_configurations[config_name]
                    components = scores['component_scores']
                    alpha_scores.append(components['communication'])
                    beta_scores.append(components['threading'])
                    gamma_scores.append(components['locality'])
                    delta_scores.append(components['gpu'])
                    epsilon_scores.append(components['openmp'])
                    actual_heuristic_scores.append(scores['composite_score'])
                else:
                    # Fallback if scorer data missing
                    alpha_scores.append(0.0)
                    beta_scores.append(0.0)
                    gamma_scores.append(0.0)
                    delta_scores.append(0.0)
                    epsilon_scores.append(0.0)
                    actual_heuristic_scores.append(0.0)
            
            # Add real component scores to DataFrame
            df['alpha_comm'] = alpha_scores          # Communication efficiency
            df['beta_thread'] = beta_scores          # Thread efficiency
            df['gamma_locality'] = gamma_scores      # Memory locality
            df['delta_gpu'] = delta_scores           # GPU utilization
            df['epsilon_openmp'] = epsilon_scores    # OpenMP efficiency
            df['heuristic_score'] = actual_heuristic_scores  # Use ACTUAL score from scorer
            df['heuristic_score_10'] = (df['heuristic_score'] * 10.0).clip(0.0, 10.0)
            
            # Check if locality is working (all non-zero)
            locality_working = any(g > 0.01 for g in gamma_scores)
            
            # Calculate derived metrics
            df['switches_per_sec'] = df['sched_events'] / df['runtime_sec'].replace(0, 1)
            df['events_per_sec'] = df['osrt_events'] / df['runtime_sec'].replace(0, 1)
            
            # Calculate simple performance scores for reference
            max_runtime = df['runtime_sec'].max()
            min_runtime = df['runtime_sec'].min()
            
            if max_runtime > min_runtime:
                df['runtime_score'] = 1 - (df['runtime_sec'] - min_runtime) / (max_runtime - min_runtime + 0.001)
            else:
                df['runtime_score'] = 1.0
            
            max_throughput = df['throughput_gflops'].max()
            df['throughput_score'] = df['throughput_gflops'] / (max_throughput + 0.001) if max_throughput > 0 else 0
            
            max_ranks = df['mpi_ranks'].max()
            df['comm_score'] = 1 - (df['mpi_ranks'] - 1) / (max_ranks - 1 + 0.001) if max_ranks > 1 else 1.0
            
            max_sps = df['switches_per_sec'].max()
            min_sps = df['switches_per_sec'].min()
            if max_sps > min_sps:
                df['efficiency_score'] = 1 - (df['switches_per_sec'] - min_sps) / (max_sps - min_sps + 0.001)
            else:
                df['efficiency_score'] = 1.0
            
            df['scalability_score'] = (df['omp_threads'] / 48.0) * df['runtime_score']
            
            # Calculate speedup
            df['speedup'] = max_runtime / df['runtime_sec'].replace(0, 1)
            
            # Sort by heuristic score
            df = df.sort_values('heuristic_score', ascending=False).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
            
            # Reorder columns for clarity (include ACTUAL heuristic components)
            column_order = [
                'rank', 'config_name', 'mpi_ranks', 'omp_threads', 'total_cores',
                'runtime_sec', 'throughput_gflops', 'speedup', 'heuristic_score', 'heuristic_score_10',
                'alpha_comm', 'beta_thread', 'gamma_locality', 'delta_gpu', 'epsilon_openmp',
                'runtime_score', 'efficiency_score', 'scalability_score', 'comm_score',
                'switches_per_sec', 'events_per_sec', 'sched_events',
                'osrt_events', 'num_rank_profiles', 'job_id'
            ]
            df = df[column_order]
            
            # Save CSV and JSON
            csv_path = output_dir / 'configuration_scores.csv'
            df.to_csv(csv_path, index=False, float_format='%.4f')
            logger.info(f"Saved scores to: {csv_path}")
            
            json_path = output_dir / 'results.json'
            try:
                from generate_results import build_experiment_provenance

                _effective_cpu_only = bool(getattr(self, "cpu_only", True)) or bool(
                    getattr(self, "_gpu_scoring_fallback", False)
                )
                _prov = build_experiment_provenance(
                    Path(self.work_directory),
                    enable_locality=locality_working,
                    cpu_only=_effective_cpu_only,
                )
                _prov["legacy_master_tuner_dashboard"] = True
            except Exception as _e:
                logger.debug(f"provenance bundle skipped: {_e}")
                _prov = {
                    "schema_version": "1.0",
                    "results_json_layout": "wrapped_v1",
                    "legacy_master_tuner_dashboard": True,
                }
            with open(json_path, 'w') as f:
                json.dump(
                    {"provenance": _prov, "configurations": df.to_dict('records')},
                    f,
                    indent=2,
                    default=str,
                )
            
            # Generate visualizations
            # Use compatible matplotlib style (works across versions)
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                try:
                    plt.style.use('seaborn-whitegrid')  # Older matplotlib
                except:
                    plt.style.use('default')  # Fallback to default
            
            # Dashboard (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('MPI+OpenMP Auto-Tuning Results Dashboard', fontsize=14, fontweight='bold')
            
            # Runtime
            ax1 = axes[0, 0]
            colors = ['#27ae60' if i == 0 else '#3498db' for i in range(len(df))]
            bars = ax1.bar(df['config_name'], df['runtime_sec'], color=colors, edgecolor='black')
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Runtime (seconds)')
            ax1.set_title('Runtime by Configuration', fontweight='bold')
            for bar, val in zip(bars, df['runtime_sec']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}s',
                        ha='center', va='bottom', fontsize=9)
            
            # Heuristic Score
            ax2 = axes[0, 1]
            colors = ['#f39c12' if i == 0 else '#95a5a6' for i in range(len(df))]
            bars = ax2.bar(df['config_name'], df['heuristic_score_10'], color=colors, edgecolor='black')
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Heuristic (0–10)')
            ax2.set_title('Heuristic Score by Configuration', fontweight='bold')
            ax2.set_ylim(0, 11)
            
            # Speedup
            ax3 = axes[1, 0]
            colors = ['#e74c3c' if i == 0 else '#95a5a6' for i in range(len(df))]
            bars = ax3.bar(df['config_name'], df['speedup'], color=colors, edgecolor='black')
            ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Configuration')
            ax3.set_ylabel('Speedup (vs worst)')
            ax3.set_title('Speedup Comparison', fontweight='bold')
            for bar, val in zip(bars, df['speedup']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}x',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Score Components (ACTUAL HEURISTIC COMPONENTS)
            ax4 = axes[1, 1]
            x = np.arange(len(df))
            
            # Detect which metrics actually vary (std > 0.01). CPU-only or GPU fallback: omit δ GPU.
            _co = bool(getattr(self, "cpu_only", False)) or bool(
                getattr(self, "_gpu_scoring_fallback", False)
            )
            delta_varies = (not _co) and (df["delta_gpu"].std() > 0.01)
            epsilon_varies = df["epsilon_openmp"].std() > 0.01
            
            # Build list of metrics to show
            metrics_to_show = []
            metrics_to_show.append(('alpha_comm', 'α Comm', '#3498db'))
            metrics_to_show.append(('beta_thread', 'β Thread', '#9b59b6'))
            
            # Add locality or alternative
            if locality_working:
                metrics_to_show.append(('gamma_locality', 'γ Locality', '#27ae60'))
                logger.info("Locality metric is working - showing in visualization")
            else:
                # Calculate MPI comm fraction as alternative to locality
                mpi_comm_frac = []
                for config_name in df['config_name']:
                    if config_name in self.scorer.scored_configurations:
                        frac = self.scorer.scored_configurations[config_name]['detailed_breakdown']['mpi_comm_fraction']
                        mpi_comm_frac.append(1.0 - frac)  # Invert so higher is better
                    else:
                        mpi_comm_frac.append(0.0)
                df['comp_comm_ratio'] = mpi_comm_frac
                metrics_to_show.append(('comp_comm_ratio', 'Comp/Comm', '#27ae60'))
                logger.info("Locality not available - showing Computation/Communication ratio instead")
            
            # Only add GPU and OpenMP if they vary (never δ in CPU-only mode)
            if delta_varies:
                metrics_to_show.append(("delta_gpu", "δ GPU", "#f39c12"))
                logger.info("GPU metric varies - showing in visualization")
            elif _co:
                logger.info("CPU-only mode: omitting δ GPU from dashboard and heatmap")
            else:
                logger.info(
                    f"GPU metric doesn't vary (all ~{df['delta_gpu'].mean():.3f}) - hiding from plot"
                )
            
            if epsilon_varies:
                metrics_to_show.append(('epsilon_openmp', 'ε OpenMP', '#e74c3c'))
                logger.info("OpenMP metric varies - showing in visualization")
            else:
                logger.info(f"OpenMP metric doesn't vary (all ~{df['epsilon_openmp'].mean():.3f}) - hiding from plot")
            
            # Plot only varying metrics
            num_metrics = len(metrics_to_show)
            width = 0.8 / num_metrics if num_metrics > 0 else 0.2
            offset = -(num_metrics - 1) / 2
            
            for i, (col, label, color) in enumerate(metrics_to_show):
                position = x + (offset + i) * width
                ax4.bar(position, df[col], width, label=label, color=color, edgecolor='black')
            
            ax4.set_xlabel('Configuration', fontweight='bold')
            ax4.set_ylabel('Component Score (0-1)', fontweight='bold')
            ax4.set_title(f'Heuristic Score Components ({num_metrics} active metrics)', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(df['config_name'], rotation=0)
            ax4.legend(loc='upper right', fontsize=9)
            ax4.set_ylim(0, 1.1)
            ax4.grid(axis='y', alpha=0.3)
            
            _plt_tight_layout()
            dashboard_path = output_dir / 'auto_tuning_dashboard.png'
            plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved dashboard: {dashboard_path}")
            plt.close()
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Build heatmap columns dynamically based on which metrics vary
            heatmap_cols = ['alpha_comm', 'beta_thread']
            heatmap_labels = ['α Comm', 'β Thread']
            
            # Add locality if working
            if locality_working:
                heatmap_cols.append('gamma_locality')
                heatmap_labels.append('γ Locality')
            
            # Only add GPU and OpenMP if they vary
            if delta_varies:
                heatmap_cols.append('delta_gpu')
                heatmap_labels.append('δ GPU')
            
            if epsilon_varies:
                heatmap_cols.append('epsilon_openmp')
                heatmap_labels.append('ε OpenMP')
            
            # Always add final heuristic score
            heatmap_cols.append('heuristic_score')
            heatmap_labels.append('HEURISTIC')
            
            heatmap_data = df.set_index('config_name')[heatmap_cols]
            heatmap_data.columns = heatmap_labels
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                        linewidths=1, ax=ax, vmin=0, vmax=1,
                        cbar_kws={'label': 'Score (0-1)'}, annot_kws={'size': 11})
            ax.set_title(f'Configuration Scores Heatmap ({len(heatmap_cols)-1} metrics + heuristic)', 
                        fontsize=14, fontweight='bold')
            
            _plt_tight_layout()
            heatmap_path = output_dir / 'scores_heatmap.png'
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved heatmap: {heatmap_path}")
            plt.close()
            
            # Profiling Metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Profiling Metrics by Configuration', fontsize=14, fontweight='bold')
            
            axes[0].bar(df['config_name'], df['sched_events'], color='#9b59b6', edgecolor='black')
            axes[0].set_xlabel('Configuration')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Scheduling Events', fontweight='bold')
            
            axes[1].bar(df['config_name'], df['events_per_sec'], color='#e74c3c', edgecolor='black')
            axes[1].set_xlabel('Configuration')
            axes[1].set_ylabel('Events/sec')
            axes[1].set_title('OSRT Events per Second', fontweight='bold')
            
            axes[2].bar(df['config_name'], df['switches_per_sec'], color='#27ae60', edgecolor='black')
            axes[2].set_xlabel('Configuration')
            axes[2].set_ylabel('Switches/sec')
            axes[2].set_title('Context Switches per Second', fontweight='bold')
            
            _plt_tight_layout()
            profiling_path = output_dir / 'profiling_metrics.png'
            plt.savefig(profiling_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved profiling metrics: {profiling_path}")
            plt.close()
            
            # Print summary
            best = df.iloc[0]
            logger.info(f"\n{'='*60}")
            logger.info(f"🏆 BEST CONFIGURATION: {best['config_name']}")
            logger.info(f"   Runtime: {best['runtime_sec']:.2f}s | Throughput: {best['throughput_gflops']:.2f} GFLOPS")
            logger.info(
                f"   Speedup: {best['speedup']:.1f}x | Heuristic: {best['heuristic_score_10']:.2f}/10 "
                f"(raw={best['heuristic_score']:.4f})"
            )
            logger.info(f"   Results saved to: {output_dir}")
            logger.info(f"{'='*60}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _call_generate_results_script(self) -> Optional[Path]:
        """
        Call generate_results.py script to ensure plots are generated with latest logic.
        This is a fallback to ensure plots are always generated even if _generate_comprehensive_visualizations fails.
        """
        try:
            # Import generate_results functions directly
            import sys
            generate_results_path = _SCRIPTS_DIR / "generate_results.py"
            
            if not generate_results_path.exists():
                logger.warning(f"generate_results.py not found at {generate_results_path}")
                return None
            from generate_results import (
                apply_cpu_only_when_gpu_absent,
                collect_experiment_data,
                generate_plots,
            )
            from autotuner.core.configuration_generator import detect_hardware_topology
            
            # Detect hardware
            try:
                hardware = detect_hardware_topology(
                    system_name=self.system_name,
                    slurm_partition=os.environ.get("HPC_PARTITION"),
                )
                total_cores = hardware.total_cores
            except:
                total_cores = self.hardware.total_cores if hasattr(self, 'hardware') else 48
            
            # Check application profile
            enable_locality = self.enable_likwid if hasattr(self, 'enable_likwid') else False
            cpu_only = bool(getattr(self, "cpu_only", True)) or bool(
                getattr(self, "_gpu_scoring_fallback", False)
            )

            # Collect data
            logger.info(f"Collecting experiment data from {self.work_directory}...")
            all_metrics = collect_experiment_data(self.work_directory, cpu_only=cpu_only)
            cpu_only, all_metrics = apply_cpu_only_when_gpu_absent(
                self.work_directory, all_metrics, cpu_only
            )
            
            if not all_metrics:
                logger.warning("No metrics collected by generate_results.py")
                return None

            _with_runtime = [
                m for m in all_metrics
                if float(m.get("runtime_sec") or m.get("total_runtime") or 0.0) > 0.0
            ]
            if not _with_runtime:
                logger.warning(
                    "generate_results.py returned metrics but none have runtime > 0; skipping plots."
                )
                return None
            
            # Generate plots
            logger.info("Generating plots using generate_results.py...")
            output_dir = generate_plots(
                self.work_directory, _with_runtime, total_cores, enable_locality, cpu_only
            )
            
            if output_dir:
                logger.info(
                    f"✅ Plots generated for {len(_with_runtime)} config(s): {output_dir}"
                )
                return output_dir
            else:
                logger.warning("generate_plots() returned None")
                return None
                
        except Exception as e:
            logger.error(f"Failed to call generate_results.py: {e}")
            import traceback
            traceback.print_exc()
            return None


    def run_scaling_tests(self, 
                         executable_path: str,
                         base_arguments: str,
                         problem_sizes: List[int],
                         core_counts: Optional[List[int]] = None,
                         time_limit: str = "02:00:00") -> Dict[str, Any]:
        """
        Run weak and strong scaling tests
        
        Args:
            executable_path: Path to application executable
            base_arguments: Base command-line arguments
            problem_sizes: List of problem sizes for weak scaling
            core_counts: List of core counts for strong scaling (auto-detected if None)
            time_limit: Time limit for each job
            
        Returns:
            Dictionary with scaling test results
        """
        logger.info("Running scaling tests...")
        
        if core_counts is None:
            # Auto-detect core counts: use powers of 2 up to total cores
            max_cores = self.hardware.total_cores
            core_counts = [2**i for i in range(1, int(max_cores.bit_length())) if 2**i <= max_cores]
        
        scaling_results = {
            'weak_scaling': {},
            'strong_scaling': {},
            'best_config': self.experiment_results.get('best_configuration')
        }
        
        # Weak scaling: keep problem size per core constant
        logger.info(f"Weak scaling test with problem sizes: {problem_sizes}")
        best_config = scaling_results['best_config']
        if best_config:
            best_mpi = best_config['details'].get('mpi_ranks', 1)
            best_omp = best_config['details'].get('omp_threads', 1)
            
            for size in problem_sizes:
                # Scale problem size with core count
                total_cores_used = best_mpi * best_omp
                scaled_size = size * total_cores_used
                args = f"{base_arguments} --size {scaled_size}"
                
                # Run job
                config = self.config_generator._create_config(
                    mpi_ranks=best_mpi,
                    omp_threads=best_omp,
                    binding="close",
                    placement="cores",
                    numa_policy="first-touch",
                    description=f"Weak scaling test: {size}"
                )
                
                job_results = self._execute_applications(
                    [config], executable_path, args, time_limit
                )
                
                if job_results:
                    performance_data = self._extract_performance_metrics(job_results)
                    if performance_data:
                        scaling_results['weak_scaling'][size] = {
                            'problem_size': scaled_size,
                            'cores': total_cores_used,
                            'metrics': list(performance_data.values())[0]
                        }
        
        # Strong scaling: keep problem size constant, vary cores
        logger.info(f"Strong scaling test with core counts: {core_counts}")
        base_size = problem_sizes[0] if problem_sizes else 1024
        
        for cores in core_counts:
            # Find MPI/OMP split that uses these cores
            # Prefer balanced configurations
            mpi_ranks = int(cores ** 0.5)
            omp_threads = cores // mpi_ranks
            
            config = self.config_generator._create_config(
                mpi_ranks=mpi_ranks,
                omp_threads=omp_threads,
                binding="close",
                placement="cores",
                numa_policy="first-touch",
                description=f"Strong scaling test: {cores} cores"
            )
            
            args = f"{base_arguments} --size {base_size}"
            job_results = self._execute_applications(
                [config], executable_path, args, time_limit
            )
            
            if job_results:
                performance_data = self._extract_performance_metrics(job_results)
                if performance_data:
                    metrics = list(performance_data.values())[0]
                    scaling_results['strong_scaling'][cores] = {
                        'mpi_ranks': mpi_ranks,
                        'omp_threads': omp_threads,
                        'time_to_solution': metrics.get('total_runtime', 0.0),
                        'speedup': scaling_results['strong_scaling'].get(
                            core_counts[0], {}
                        ).get('time_to_solution', 1.0) / max(metrics.get('total_runtime', 1.0), 0.001),
                        'efficiency': 0.0  # Will calculate relative to first run
                    }
        
        # Calculate efficiency for strong scaling
        if scaling_results['strong_scaling']:
            baseline_time = scaling_results['strong_scaling'][core_counts[0]]['time_to_solution']
            baseline_cores = core_counts[0]
            
            for cores, data in scaling_results['strong_scaling'].items():
                if baseline_time > 0:
                    ideal_speedup = cores / baseline_cores
                    actual_speedup = baseline_time / max(data['time_to_solution'], 0.001)
                    data['efficiency'] = actual_speedup / ideal_speedup if ideal_speedup > 0 else 0.0
        
        logger.info(f"Scaling tests completed: {len(scaling_results['weak_scaling'])} weak scaling, {len(scaling_results['strong_scaling'])} strong scaling")
        return scaling_results
    
    def print_summary(self) -> None:
        """Print a summary of the auto-tuning experiment"""
        print("\n" + "="*80)
        print("MPI+OpenMP Auto-Tuning Experiment Summary")
        print("="*80)
        
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Application: {self.application_name}")
        print(f"System: {self.hardware.system_name}")
        print(f"Hardware: {self.hardware.total_cores} cores, {self.hardware.sockets} sockets, {self.hardware.numa_domains} NUMA domains")
        print(f"Working Directory: {self.work_directory}")
        
        best = self.experiment_results.get('best_configuration')
        _best_rt = 0.0
        if best and isinstance(best.get('details'), dict):
            try:
                _best_rt = float(best['details'].get('runtime_sec') or 0.0)
            except (TypeError, ValueError):
                _best_rt = 0.0
        # Load fastest + heuristic picks from generate_results JSON when available
        fastest_name = None
        fastest_rt = 0.0
        final_json_path = self.work_directory / "comprehensive_results" / "results.json"
        if final_json_path.exists():
            try:
                with open(final_json_path, "r") as f:
                    final_res = json.load(f)
                rows = final_res.get("configurations", final_res)
                if isinstance(rows, list):
                    valid = [r for r in rows if float(r.get("runtime_sec") or 0) > 0]
                    if valid:
                        fast_row = min(valid, key=lambda r: float(r["runtime_sec"]))
                        fastest_name = fast_row.get("config_name")
                        fastest_rt = float(fast_row["runtime_sec"])
            except Exception:
                pass

        if best and _best_rt > 0.0:
            print(f"\n🎯 Best Heuristic Configuration: {best['name']}")
            _raw = float(best.get("score", 0.0) or 0.0)
            _det = best.get("details") or {}
            try:
                _s10 = float(_det.get("heuristic_score_10", _raw * 10.0))
            except (TypeError, ValueError):
                _s10 = _raw * 10.0
            print(f"📊 Heuristic Score: {_s10:.2f}/10 (raw composite {_raw:.4f})")
            if 'details' in best and best['details']:
                print(f"⚙️  MPI Ranks: {best['details'].get('mpi_ranks', 'N/A')}")
                print(f"🧵 OpenMP Threads: {best['details'].get('omp_threads', 'N/A')}")
        elif self.experiment_results.get('configurations_tested', 0) > 0:
            print("\n⚠️  No best configuration — all jobs failed or produced no measurable runtime.")

        if fastest_name and fastest_rt > 0:
            print(f"\n⚡ Fastest Runtime: {fastest_name} ({fastest_rt:.4f}s wall clock)")
            if best and best.get('name') and fastest_name != best.get('name'):
                print(
                    f"   (Heuristic pick '{best['name']}' differs — use fastest for raw speed, "
                    f"heuristic for hybrid MPI+GPU balance.)"
                )

        print(f"\n📈 Results Summary:")
        if self.profile_subset and 'phase1_configs' in self.experiment_results:
            print(f"  Phase-1 (cheap run) configs: {self.experiment_results['phase1_configs']}")
        print(f"  Phase-2 SLURM jobs (last batch): {self.experiment_results['configurations_tested']}")
        print(f"  Configurations with metrics (scored): {self.experiment_results['successful_runs']}")
        improvement = float(self.experiment_results.get('performance_improvement') or 0.0)
        speedup = float(self.experiment_results.get('speedup') or 1.0)
        num_successful = self.experiment_results.get('successful_runs', 0)
        best_rt = float(self.experiment_results.get('best_runtime') or fastest_rt or 0.0)
        worst_rt = float(self.experiment_results.get('worst_runtime') or 0.0)

        if num_successful <= 1:
            print(f"  Runtime spread: N/A (only {num_successful} successful configuration)")
        elif best_rt > 0 and worst_rt > 0:
            ratio = worst_rt / best_rt
            fast_lbl = fastest_name or "fastest"
            if ratio > 100.0:
                print(
                    f"  Runtime spread: slowest is {ratio:.0f}× slower than {fast_lbl} "
                    f"({best_rt:.4f}s vs {worst_rt:.2f}s)"
                )
            elif ratio > 1.05:
                print(
                    f"  Runtime spread: slowest is {(ratio - 1.0) * 100.0:.1f}% slower "
                    f"than {fast_lbl} ({ratio:.2f}×)"
                )
            else:
                print(f"  Runtime spread: runtimes similar across configs ({ratio:.2f}×)")
        else:
            print("  Runtime spread: see comprehensive_results/results.json")

        
        print(f"\n📋 Next Steps:")
        print(f"  1. Review detailed report in: {self.work_directory}")
        print(f"  2. Apply optimal configuration to production jobs")
        print(f"  3. Monitor performance improvements")
        print(f"  4. Consider running scaling analysis")


def _parse_nsight_trace_arg(value: str) -> str:
    t = str(value).strip()
    if not re.fullmatch(r'[a-zA-Z0-9_,]+', t):
        raise argparse.ArgumentTypeError(
            "comma-separated Nsight domains only, e.g. cuda,cublas or osrt,mpi,cuda,cublas"
        )
    return t


def main():
    """Main entry point for the master auto-tuner"""
    parser = argparse.ArgumentParser(
        description="Master Auto-Tuner for MPI+OpenMP Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # mpiP prefix (directory containing lib/libmpiP.so) — pass on compute-visible path, e.g. $HOME/software/mpiP
  # For MVAPICH2 jobs, build mpiP against the same MPI module you load in the job, not a different vendor.

  # Run complete auto-tuning for sparse matrix application (with mpiP for α)
  python master_tuner.py --application sparse_matrix --account TG-ABC123 \\
                         --mpip-path "$HOME/software/mpiP"

  # Run pilot study (short runs for testing)
  python master_tuner.py --application sparse_matrix --pilot-only --account TG-ABC123 \\
                         --mpip-path "$HOME/software/mpiP"

  # Multi-node execution (2 nodes)
  python master_tuner.py --application sparse_test --account cs --nodes 2 \\
                         --executable $(pwd)/applications/sparse_application \\
                         --arguments " --size 4096 --iterations 30" \\
                         --mpip-path "$HOME/software/mpiP"

  # LULESH (build: cd applications && make lulesh2.0)
  python master_tuner.py --application lulesh --application-profile lulesh --account YOUR_ACCOUNT \\
                         --nodes 2 --time-limit 02:00:00 \\
                         --mpip-path "$HOME/software/mpiP"

  # Custom time limit
  python master_tuner.py --application hybrid_vec --application-profile hybrid_vec \\
                         --executable "$(pwd)/applications/HybridVec/hybrid_vec_gpu" \\
                         --account TG-ABC123 --time-limit 04:00:00 \\
                         --mpip-path "$HOME/software/mpiP"
        """
    )
    
    parser.add_argument(
        '--application', required=True,
        help='Name of the application to auto-tune'
    )
    
    parser.add_argument(
        '--application-profile',
        choices=sorted(APPLICATION_PROFILES.keys()),
        default=None,
        help=(
            "Application workload profile for auto-configuration (see autotuner.app_registry.profiles.APPLICATION_PROFILES). "
            "Overrides --cpu-only / --enable-likwid when the profile sets them."
        ),
    )
    
    parser.add_argument(
        '--account', required=True,
        help='HPC allocation account (e.g., TG-ABC123)'
    )
    
    parser.add_argument(
        '--system', default=None,
        help='HPC system name for config lookup (optional, uses auto-detection if not specified)'
    )
    
    parser.add_argument(
        '--partition', default=None,
        help='SLURM partition to use (overrides system defaults and auto-detection)'
    )

    parser.add_argument(
        '--cpus-per-node',
        type=int,
        default=None,
        metavar='N',
        help=(
            "Max CPUs per node for MPI×OpenMP layout generation. Caps lscpu when SLURM cannot "
            "satisfy a full node (e.g. Vista 'gh' often needs 72 though lscpu shows 144). "
            "Sets HPC_SLURM_CPUS_PER_NODE for this run."
        ),
    )
    
    parser.add_argument(
        '--executable', default='/path/to/your/application',
        help='Path to the application executable'
    )
    
    parser.add_argument(
        '--arguments', default='--size 1024 --iterations 100',
        help='Command line arguments for the application'
    )
    
    parser.add_argument(
        '--phase1-args', dest='phase1_arguments', default=None,
        help='Explicit arguments for Phase-1 preliminary screening (overrides automatic lightweight logic)'
    )
    parser.add_argument(
        '--phase2-args', dest='phase2_arguments', default=None,
        help='CLI for Phase-2 profiling jobs (Nsight + LIKWID + Run 1a). '
             'Apps using GPU light-size cap: default matches Phase-1 workload (see logs); '
             'set this to match full --arguments for production-sized traces.'
    )

    parser.add_argument(
        '--adaptive-weights', action='store_true',
        help='Enable adaptive heuristic weights (rule-based, no ML/AI). Weights adjust based on bottleneck detection and performance analysis.'
    )
    
    # Custom action so --enable-likwid is not overridden by profile (sparse_matrix has enable_likwid: False)
    class StoreTrueWithExplicit(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs=0, default=False, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, True)
            setattr(namespace, self.dest + '_set', True)
    parser.add_argument(
        '--enable-likwid', action=StoreTrueWithExplicit, dest='enable_likwid',
        help='Enable LIKWID profiling for memory locality metrics (requires LIKWID). Overrides profile default.'
    )
    parser.add_argument(
        '--likwid-path', default=None, metavar='PATH',
        help='Path to LIKWID install so compute nodes find it (e.g. $HOME/software/likwid). No hardcoding in repo.'
    )
    parser.add_argument(
        '--enable-mpip', action='store_true', default=True,
        help='Enable mpiP for α (communication) when libmpiP.so is found (PMPI-level timing). Default: on.'
    )
    parser.add_argument(
        '--no-mpip', action='store_false', dest='enable_mpip',
        help='Disable mpiP (do not LD_PRELOAD libmpiP.so).'
    )
    parser.add_argument(
        '--mpip-path', default=None, metavar='PATH',
        help='Path to mpiP install so compute nodes find libmpiP.so (e.g. $HOME/software/mpiP). '
             'Checked first when set. If no .mpiP files appear: ensure this path is visible on compute nodes, '
             'use mpiP 3.5+ with lib/libmpiP.so, and check SLURM stdout for "mpiP:" and any "Could not open" message.'
    )
    parser.add_argument(
        '--no-mpip-srun-run1a', action='store_true', default=False,
        help='Disable MVAPICH2 mpiP via srun+LD_PRELOAD on Run 1a (falls back to mpirun without mpiP preload). '
             'Use if your site srun/PMI rejects this path; tune hpc_config srun_mpi_flag otherwise.',
    )
    # Custom action for cpu-only to track if user explicitly set it
    class StoreTrueCpuOnly(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs=0, default=False, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, True)
            setattr(namespace, 'cpu_only_set', True)

    parser.add_argument(
        '--cpu-only', action=StoreTrueCpuOnly, dest='cpu_only',
        help='CPU-only mode: redistributes GPU and OpenMP weights to comm/thread metrics (default: False, unless profile overrides)'
    )
    
    parser.add_argument(
        '--time-limit', default='02:00:00',
        help='Time limit for each job in HH:MM:SS format (default: 02:00:00)'
    )
    
    parser.add_argument(
        '--profile-subset', action='store_true',
        help=(
            'Two-phase: Phase-1 prescreen then Phase-2 full profiling. '
            'Default: Phase-2 only runs configs that completed Phase-1 with a valid runtime (subset). '
            'Use --phase2-all-configs to profile every layout even when Phase-1 failed (debugging / comparison).'
        ),
    )
    parser.add_argument(
        '--phase2-all-configs',
        dest='phase2_all_configs',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'With --profile-subset: profile all generated configs in Phase-2 (wastes GPU time if some Phase-1 jobs failed). '
            'Default off: Phase-2 uses only the heuristic subset of Phase-1 successes.'
        ),
    )
    parser.add_argument(
        '--skip-phase1', action='store_true',
        help='Skip Phase-1; run full profiling for all configs (use when Phase-1 times out or is slow). Implies full profiling for all 6 configs.'
    )
    parser.add_argument(
        '--no-nsight', action='store_true',
        help='Do not use Nsight Systems: run application only and score from runtime/throughput. Much faster (hours not days); α–ε use runtime/log-based estimates where trace data is absent.'
    )
    parser.add_argument(
        '--multi-rank-profile', action='store_true',
        help='Enable Nsight profiling for multi-rank jobs (Run 2: nsys wraps full mpirun after Run 1). Default: multi-rank is Run 1 only (no Nsight). Single-rank always uses Nsight when profiling is on.'
    )
    parser.add_argument(
        '--nsight-duration', type=int, default=None, metavar='SEC',
        help='Limit Nsight trace to SEC seconds (faster profiling, smaller .nsys-rep; α/β from a slice). E.g. 60 or 120. Omit for full run.'
    )
    parser.add_argument(
        '--nsight-trace', type=_parse_nsight_trace_arg, default=None, metavar='DOMAINS',
        help=(
            'Nsight Systems --trace=DOMAINS (default: osrt,mpi,cuda,cublas). '
            'Faster SU: cuda,cublas — less collector overhead; α usually still from mpiP when enabled. '
            'Use full default when you need OSRT/MPI columns in the trace for extractor fallbacks.'
        )
    )
    parser.add_argument(
        '--deep-profile-finalists', action='store_true',
        help='After the main profiling batch, pick up to N finalists (fastest runtime, best heuristic, median runtime) '
             'and submit extra jobs with Nsight on MPI ranks 0, N/2, N-1 for better multi-rank coverage (cost-bounded). '
             'Requires multi-rank profiling; merges job IDs into phase2_job_ids.json for generate_results.'
    )
    parser.add_argument(
        '--deep-profile-max-configs', type=int, default=3, metavar='N',
        help='Max finalist configs for --deep-profile-finalists (default: 3).'
    )
    parser.add_argument(
        '--deep-profile-time-limit', default=None, metavar='HH:MM:SS',
        help='Time limit for deep finalist jobs (default: same as --time-limit).'
    )
    parser.add_argument(
        '--phase1-time-limit', default=None, metavar='HH:MM:SS',
        help='Time limit for Phase-1 jobs (default: 00:20:00 when --profile-subset). Use e.g. 00:15:00 for cheaper runs.'
    )
    parser.add_argument(
        '--phase2-time-limit', default=None, metavar='HH:MM:SS',
        help=(
            'Time limit for profiling jobs (Phase-2 / full Nsight+LIKWID with Run 2). '
            'Defaults to --time-limit if omitted. Multi-node Nsight collection+export often needs 4–8+ hours; '
            'use e.g. 06:00:00 or 08:00:00 if jobs die with TIME LIMIT during Run 2.'
        )
    )
    parser.add_argument(
        '--work-dir',
        help='Working directory for experiments (default: auto-generated)'
    )
    
    parser.add_argument(
        '--nodes', type=int, default=1,
        help='Number of nodes to use for multi-node execution (default: 1)'
    )
    
    parser.add_argument(
        '--pilot-only', action='store_true',
        help='Run only recommended configurations for testing'
    )
    
    parser.add_argument(
        '--scaling-test', action='store_true',
        help='Run weak and strong scaling tests after auto-tuning'
    )
    
    parser.add_argument(
        '--scaling-problem-sizes', nargs='+', type=int,
        default=[1024, 2048, 4096],
        help='Problem sizes for weak scaling test (default: 1024 2048 4096)'
    )
    
    parser.add_argument(
        '--scaling-cores', nargs='+', type=int,
        default=None,
        help='Core counts for strong scaling test (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--env-path', type=str, default=None,
        help='Additional directories to prepend to PATH in job scripts (colon-separated). '
             'Use this to ensure nsys/likwid are found on compute nodes. '
             'Example: --env-path "/opt/apps/cuda/11.4/bin:/path/to/likwid/bin"'
    )
    
    args = parser.parse_args()

    apply_application_cli_defaults(args)

    profile_key = resolve_profile_key(args.application, args.application_profile)

    # Apply profile configuration
    if profile_key != 'custom' and profile_key in APPLICATION_PROFILES:
        profile = APPLICATION_PROFILES[profile_key]
        # Override flags based on profile (unless user explicitly set them)
        if not hasattr(args, 'cpu_only_set') and profile['cpu_only'] is not None:
            args.cpu_only = profile['cpu_only']
        if not hasattr(args, 'enable_likwid_set') and profile['enable_likwid'] is not None:
            args.enable_likwid = profile['enable_likwid']
        
        print(f"📋 Using application profile: {profile['description']}")
        print(f"   CPU-only mode: {args.cpu_only}")
        print(f"   Locality (LIKWID): {args.enable_likwid}")
        print()
    else:
        print(f"📋 Using custom configuration")
        print(f"   CPU-only mode: {args.cpu_only}")
        print(f"   Locality (LIKWID): {args.enable_likwid}")
        print()

    # Smart executable path resolution (see autotuner.app_registry.application_resources)
    if args.executable == '/path/to/your/application':
        root_dir = Path(__file__).resolve().parent.parent
        applications_dir = root_dir / "applications"
        candidate = resolve_default_executable_path(applications_dir, args.application)
        if candidate is not None:
            if candidate.exists():
                print(f"✅ Auto-detected application executable: {candidate}")
                args.executable = str(candidate)
            else:
                print(f"⚠️  Application executable not found at: {candidate}")
                print(f"   Please run 'make' in {applications_dir}")
                print(f"   Or provide full path with --executable")
                sys.exit(1)
    # Use logical path (/home/...) in scripts; LEAP2 resolves to /mmfs1/home/... but users see /home
    args.executable = str(normalize_to_logical_path(args.executable))
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set partition override if specified
    if args.partition:
        os.environ['HPC_PARTITION'] = args.partition
        print(f"📋 Overriding SLURM partition: {args.partition}")

    if args.cpus_per_node is not None and args.cpus_per_node > 0:
        os.environ['HPC_SLURM_CPUS_PER_NODE'] = str(args.cpus_per_node)
        print(
            f"📋 SLURM layout cap: HPC_SLURM_CPUS_PER_NODE={args.cpus_per_node} "
            f"(MPI×OpenMP product per node will not exceed this)"
        )
    
    try:
        # Create master auto-tuner (uses MPI backend)
        tuner = MasterAutoTuner(
            application_name=args.application,
            account=args.account,
            system_name=args.system,
            work_directory=Path(args.work_dir) if args.work_dir else None,
            adaptive_weights=args.adaptive_weights,
            enable_likwid=args.enable_likwid,
            likwid_path=args.likwid_path,
            enable_mpip=getattr(args, 'enable_mpip', True),
            mpip_path=getattr(args, 'mpip_path', None),
            no_mpip_srun_run1a=getattr(args, 'no_mpip_srun_run1a', False),
            cpu_only=args.cpu_only,
            num_nodes=args.nodes,
            profile_subset=getattr(args, 'profile_subset', False),
            phase2_all_configs=getattr(args, 'phase2_all_configs', False),
            skip_phase1=getattr(args, 'skip_phase1', False),
            no_nsight=getattr(args, 'no_nsight', False),
            multi_rank_profile=getattr(args, 'multi_rank_profile', True),  # Default True: Run 2 Nsight wraps full mpirun
            nsight_duration_sec=getattr(args, 'nsight_duration', None),
            nsight_trace_domains=getattr(args, 'nsight_trace', None),
            phase1_time_limit=getattr(args, 'phase1_time_limit', None),
            phase2_time_limit=getattr(args, 'phase2_time_limit', None),
            env_path=getattr(args, 'env_path', None),
            deep_profile_finalists=getattr(args, 'deep_profile_finalists', False),
            deep_profile_max_configs=getattr(args, 'deep_profile_max_configs', 3),
            deep_profile_time_limit=getattr(args, 'deep_profile_time_limit', None),
        )

        if (
            not getattr(args, "no_nsight", False)
            and getattr(args, "env_path", None)
            and not _env_path_has_executable_nsys(args.env_path)
        ):
            logger.warning(
                "Nsight profiling is enabled but --env-path does not include a directory with an "
                "executable `nsys`. Phase-2 Run 2 will be skipped on compute nodes unless `nsys` is "
                "on the default PATH (e.g. modules). Add the Nsight Systems bin directory to "
                "--env-path (alongside LIKWID if needed)."
            )
        
        # Run auto-tuning
        results = tuner.run_complete_auto_tuning(
            executable_path=args.executable,
            arguments=args.arguments,
            time_limit=args.time_limit,
            pilot_only=args.pilot_only,
            phase1_arguments=args.phase1_arguments,
            phase2_arguments=getattr(args, 'phase2_arguments', None),
        )
        
        # Run scaling tests if requested
        if args.scaling_test:
            logger.info("Running scaling tests...")
            scaling_results = tuner.run_scaling_tests(
                executable_path=args.executable,
                base_arguments=args.arguments,
                problem_sizes=args.scaling_problem_sizes,
                core_counts=args.scaling_cores,
                time_limit=args.time_limit
            )
            
            # Save scaling results
            scaling_file = tuner.work_directory / "scaling_results.json"
            with open(scaling_file, 'w') as f:
                json.dump(scaling_results, f, indent=2, default=str)
            logger.info(f"Scaling test results saved to {scaling_file}")
        
        # Print summary
        tuner.print_summary()
        
        # Exit successfully
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Auto-tuning interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Auto-tuning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Auto-purge stale bytecode caches before any imports are used.
    # On shared HPC filesystems the .pyc files can lag behind edits to .py files,
    # causing old logic to run even after fixes are applied.  Removing __pycache__
    # here forces Python to recompile every module freshly on this invocation.
    import shutil as _shutil
    _repo_root = Path(__file__).resolve().parent.parent
    for _pc in list(_repo_root.rglob("__pycache__")):
        try:
            _shutil.rmtree(_pc)
        except Exception:
            pass  # best-effort; non-fatal if a cache dir is locked or missing
    del _shutil, _repo_root
    main()


