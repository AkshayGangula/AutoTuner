#!/usr/bin/env python3
"""
SLURM Job Script Generator

Handles generation of SLURM job scripts for MPI+OpenMP auto-tuning experiments.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from autotuner.app_registry.minimd import slurm_full_runtime_fallback_from_run1_out_lines
from autotuner.utils.path_utils import experiment_artifact_dir_shell_expr, path_to_shell_home_literal

from .models import SLURMJobConfig

logger = logging.getLogger(__name__)


def _job_uses_mvapich2(job_config: SLURMJobConfig) -> bool:
    """True when job modules suggest MVAPICH2 (Hydra + OpenMPI-style flags are incompatible)."""
    for m in job_config.modules or []:
        if m and "mvapich" in m.lower():
            return True
    return False


def _coerce_additional_bool(val, default: bool = True) -> bool:
    """Parse additional_options bools (JSON may use true/false; strings '0'/'1' also seen)."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return default


class SLURMJobGenerator:
    """Generates SLURM job scripts for auto-tuning experiments"""
    
    def __init__(self, work_directory: Path, system_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the job generator
        
        Args:
            work_directory: Working directory for job scripts
            system_config: HPC system dict (``artifact_dir``, ``slurm_log_dir``, …)
        """
        self.work_directory = work_directory
        self.system_config = system_config or {}
        self.scripts_dir = work_directory / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_job_script(self,
            job_config: SLURMJobConfig,
            executable_path: str,
            arguments: str = "",
            profiling: bool = True,
            likwid_enabled: bool = False,
            likwid_path: Optional[str] = None,
            enable_mpip: bool = True,
            mpip_path: Optional[str] = None,
            light_arguments: Optional[str] = None,
            multi_rank_profile: bool = False,
            nsight_duration_sec: Optional[int] = None,
            nsight_trace_domains: Optional[str] = None,
            spread_nsight_ranks: bool = False,
            likwid_rank0_only: bool = False) -> Path:
        """
        Generate a SLURM job script for MPI backend
        
        Args:
            job_config: SLURM job configuration
            executable_path: Path to the executable
            arguments: Command line arguments (full run)
            profiling: Whether to enable profiling (Nsight Systems)
            likwid_enabled: Whether to enable LIKWID profiling
            likwid_path: Optional path to LIKWID install (e.g. $HOME/software/likwid). If set, job script exports LIKWID_PATH so compute nodes find LIKWID.
            enable_mpip: If True, try to load mpiP for α (communication) from PMPI-level timing.
            mpip_path: Optional path to mpiP install (e.g. $HOME/software/mpiP). If set, job script checks this path first for libmpiP.so.
            light_arguments: Reduced args for multi-rank (e.g. --size 256 --iterations 1). When set, Run 1 uses this; Run 2 (if enabled) also uses this for Nsight.
            multi_rank_profile: If True, multi-rank jobs run Run 2 (Nsight wraps full mpirun) after Run 1 and export to SQLite. If False, multi-rank is Run 1 only (no Nsight).
            nsight_duration_sec: If set, limit Nsight trace to this many seconds (faster profiling, smaller .nsys-rep). α/β from a representative slice.
            nsight_trace_domains: Comma list for nsys --trace=... (default osrt,mpi,cuda,cublas). Use e.g. cuda,cublas for less overhead; α often still from mpiP.
            spread_nsight_ranks: Reserved for deep finalists; multi-rank Run 2 always uses one Nsight session wrapping the full MPI launch (single profile.sqlite).
            likwid_rank0_only: If True, Run 1b/1c run likwid-perfctr only on MPI rank 0; other ranks run the app bare.
                γ uses rank-0 counters.

        Returns:
            Path to the generated job script
        """
        script_content = self._build_job_script_content(
            job_config, executable_path, arguments, profiling, likwid_enabled,
            likwid_path, enable_mpip, mpip_path, light_arguments, multi_rank_profile, nsight_duration_sec,
            nsight_trace_domains, spread_nsight_ranks, likwid_rank0_only,
        )
        
        # Save script
        script_file = self.scripts_dir / f"{job_config.job_name}.slurm"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_file.chmod(0o755)
        
        logger.info(f"Generated job script: {script_file}")
        return script_file
    
    def _build_job_script_content(self,
            job_config: SLURMJobConfig,
            executable_path: str,
            arguments: str,
            profiling: bool,
            likwid_enabled: bool,
            likwid_path: Optional[str] = None,
            enable_mpip: bool = True,
            mpip_path: Optional[str] = None,
            light_arguments: Optional[str] = None,
            multi_rank_profile: bool = False,
            nsight_duration_sec: Optional[int] = None,
            nsight_trace_domains: Optional[str] = None,
            spread_nsight_ranks: bool = False,
            likwid_rank0_only: bool = False) -> str:
        """Build the content of the SLURM job script"""
        
        # SLURM directives
        directives = self._build_directives(job_config)
        
        # Environment setup
        env_setup = self._build_environment_setup(job_config, executable_path)
        
        # Job execution (profiling handled in _build_mpi_execution)
        execution = self._build_execution_section(
            job_config, executable_path, arguments, profiling, likwid_enabled, likwid_path, enable_mpip, mpip_path, light_arguments, multi_rank_profile, nsight_duration_sec, nsight_trace_domains, spread_nsight_ranks, likwid_rank0_only,
        )
        
        # Cleanup and reporting
        cleanup = self._build_cleanup_section(profiling)
        
        # Combine all sections
        script_lines = directives + env_setup + execution + cleanup
        
        return "\n".join(script_lines)
    
    def _build_directives(self, job_config: SLURMJobConfig) -> list:
        """Build SLURM directives section"""
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_config.job_name}",
            f"#SBATCH --account={job_config.account}",
            f"#SBATCH --time={job_config.time_limit}",
            f"#SBATCH --nodes={job_config.nodes}",
            f"#SBATCH --ntasks-per-node={job_config.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={job_config.cpus_per_task}",
            # f"#SBATCH --mem={job_config.mem_per_node}", # Disabled for TACC (full node allocation)
            f"#SBATCH --partition={job_config.partition}",
            f"#SBATCH --output={job_config.output_file}",
            f"#SBATCH --error={job_config.error_file}"
        ]
        
        # Only include QOS if it's provided and not empty
        if job_config.qos and job_config.qos.strip():
            directives.append(f"#SBATCH --qos={job_config.qos}")
        
        if job_config.email:
            directives.extend([
                f"#SBATCH --mail-user={job_config.email}",
                f"#SBATCH --mail-type={job_config.mail_type}"
            ])
        
        # Pass-through #SBATCH only for real SLURM keys. These are consumed by _build_mpi_execution, not sbatch:
        _internal_additional = frozenset({"launcher", "mpip_mvapich_srun", "srun_mpi_flag"})
        for option, value in job_config.additional_options.items():
            if option in _internal_additional:
                continue
            if value == "":
                directives.append(f"#SBATCH --{option}")
            else:
                directives.append(f"#SBATCH --{option}={value}")
        
        return directives
    
    def _build_environment_setup(self, job_config: SLURMJobConfig, executable_path: str = "") -> list:
        """Build environment setup section for MPI"""
        env_setup = [
            "",
            "# Environment Setup",
            "# Batch cwd is #SBATCH --chdir (scratch); avoid cd to GPFS submit dir (ESTALE on some LEAP2 nodes).",
            ""
        ]
        
        # Do not pipe module load — subshell would skip LD_LIBRARY_PATH / INTEL_HOME in this script.
        if job_config.modules:
            env_setup.extend([
                "# Without this, `module load` is skipped silently (no FATAL)",
                "if ! type module 2>/dev/null | grep -q function; then",
                "  _lmod_sourced=\"\"",
                "  for f in /opt/apps/lmod/lmod/init/bash /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do",
                "    if [ -r \"$f\" ]; then",
                "      # shellcheck source=/dev/null",
                "      . \"$f\"",
                "      _lmod_sourced=\"$f\"",
                "      break",
                "    fi",
                "  done",
                "  if ! type module 2>/dev/null | grep -q function; then",
                "    echo \"FATAL: Lmod init not found (tried standard paths). Cannot load modules.\" >&2",
                "    exit 1",
                "  fi",
                "  echo \"Sourced Lmod from: $_lmod_sourced\"",
                "fi",
                "# Drop SLURM/node inherited stacks before our loads",
                "module purge 2>/dev/null || true",
                "",
            ])
            env_setup.append("# Load required modules (fail job if load fails — do not hide with || true)")
            for module in job_config.modules:
                # Never use 2>/dev/null || true: silent failure leaves system defaults and can cause issues.
                esc = module.replace('"', '\\"')
                env_setup.append(
                    f'module load {module} || {{ echo "FATAL: module load failed: {esc}" >&2; exit 1; }}'
                )
            env_setup.extend([
                "echo \"--- module list ---\"",
                "module list",
                "echo \"--- MPI libs (sanity) ---\"",
                "_MR=\"$(command -v mpirun 2>/dev/null || command -v mpiexec 2>/dev/null || true)\"",
                "if [ -n \"$_MR\" ] && [ -x \"$_MR\" ]; then",
                "  echo \"mpirun: $_MR\"",
                "  ldd \"$_MR\" 2>/dev/null | grep -E 'ucs|ucp|openmpi|mpi' | head -25 || true",
                "fi",
                "",
            ])
            # Ensure Intel OpenMP (libiomp5) is findable when Intel module is used (e.g. applications built with icc -qopenmp).
            # Module load may fail on compute nodes; add explicit path fallback so jobs still run.
            if any("intel" in m.lower() for m in job_config.modules):
                env_setup.extend([
                    "# Intel OpenMP (libiomp5) fallback — required for Intel-built OpenMP binaries",
                    "if ! ldconfig -p 2>/dev/null | grep -q libiomp5; then",
                    "  IOPMP_DIR=\"\"",
                    "  for d in \"/opt/intel/compiler/latest/linux/compiler/lib/intel64_lin\" \\",
                    "           \"/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin\" \\",
                    "           \"/opt/apps/intel/compiler/latest/linux/compiler/lib/intel64_lin\"; do",
                    "    if [ -f \"${d}/libiomp5.so\" ]; then",
                    "      IOPMP_DIR=\"$d\"",
                    "      break",
                    "    fi",
                    "  done",
                    "  if [ -z \"$IOPMP_DIR\" ] && [ -n \"$INTEL_HOME\" ]; then",
                    "    for d in \"$INTEL_HOME/compiler/latest/linux/compiler/lib/intel64_lin\" \\",
                    "             \"$INTEL_HOME/compiler/2024.2.0/linux/compiler/lib/intel64_lin\"; do",
                    "      if [ -f \"${d}/libiomp5.so\" ]; then",
                    "        IOPMP_DIR=\"$d\"",
                    "        break",
                    "      fi",
                    "    done",
                    "  fi",
                    "  if [ -n \"$IOPMP_DIR\" ]; then",
                    "    export LD_LIBRARY_PATH=\"$IOPMP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"",
                    "    echo \"Using Intel OpenMP from: $IOPMP_DIR\"",
                    "  fi",
                    "fi",
                    ""
                ])
            
        # Enable strict error checking after modules are loaded
        env_setup.append("set -e  # Exit on error for actual failures")
        env_setup.append("ulimit -s unlimited  # Prevent stack overflow errors")
        env_setup.append("")
        
        launcher_raw = job_config.additional_options.get("launcher", "mpirun")
        uses_mvapich = _job_uses_mvapich2(job_config)
        if "mpirun" in launcher_raw or "orterun" in launcher_raw:
            env_setup.append(
                "# MPI + CUDA configurations"
            )
            if uses_mvapich:
                # MVAPICH2 uses Hydra; OMPI_MCA_* are ignored and clutter the environment.
                env_setup.append("# MVAPICH2: skip OpenMPI MCA (not applicable).")
                env_setup.append("")
                if job_config.nodes == 1:
                    env_setup.append(
                        "# MVAPICH2 Hydra defaults to the Slurm launcher (srun). After `module purge`, srun "
                        "is often missing — use fork on single-node allocations."
                    )
                    env_setup.append("export HYDRA_LAUNCHER=fork")
                    env_setup.extend([
                        "# Silence MVAPICH2 hybrid-mode suggestions in .err files.",
                        "export MV2_USE_ALIGNED_ALLOC=1",
                        "export MV2_USE_THREAD_WARNING=0",
                        "export MV2_ENABLE_AFFINITY=0",
                        "",
                    ])
                else:
                    env_setup.extend([
                        "# Multi-node MVAPICH2: Hydra needs `srun` on PATH (load a Slurm client module in hpc_config).",
                        "if ! command -v srun >/dev/null 2>&1; then",
                        "  for _slurm_mod in slurm/slurm/21.08.8 slurm; do",
                        "    module load \"$_slurm_mod\" 2>/dev/null && break",
                        "  done",
                        "fi",
                        "command -v srun >/dev/null 2>&1 || echo \"WARNING: srun not found; MVAPICH2 multi-node launch may fail\" >&2",
                        "# Suppress MVAPICH2 startup noise: RDMA CM is not needed on Ethernet/IB-verbs clusters.",
                        "# MV2_USE_RDMA_CM=0: skip RDMA CM initialization (avoids 'RDMA CM Initialization failed' warnings).",
                        "# MV2_SUPPRESS_JOB_STARTUP_PERFORMANCE_WARNING=1: silence the homogeneous-cluster startup hint.",
                        "# MV2_USE_ALIGNED_ALLOC=1: aligned allocations for hybrid MPI+OpenMP (silences perf suggestion).",
                        "# MV2_USE_THREAD_WARNING=0: suppress 'multi-thread capability' info banner.",
                        "export MV2_USE_RDMA_CM=0",
                        "export MV2_SUPPRESS_JOB_STARTUP_PERFORMANCE_WARNING=1",
                        "export MV2_USE_ALIGNED_ALLOC=1",
                        "export MV2_USE_THREAD_WARNING=0",
                        "export MV2_ENABLE_AFFINITY=0",
                        "",
                    ])
            else:
                env_setup.append("export OMPI_MCA_opal_cuda_support=0")
                env_setup.append("")
                # OpenMPI environment variables for proper threading and binding support
                # (MV2_* variables are MVAPICH2-specific and are safely ignored by OpenMPI)
                env_setup.append("# OpenMPI threading support (enables MPI_THREAD_FUNNELED for OpenMP+MPI hybrid)")
                env_setup.append("export OMPI_MCA_mpi_thread_level=1")
                env_setup.append("export OMPI_MCA_btl_openib_allow_ib=1")
                env_setup.append("export OMPI_MCA_pml=ob1")  # Use OB1 PML for reliable Infiniband comms
                env_setup.append("")
                env_setup.append("# OpenMPI CPU binding: let mpirun --bind-to core handle binding")
                env_setup.append("export OMPI_MCA_hwloc_base_binding_policy=core")
                env_setup.append("export OMPI_MCA_rmaps_base_mapping_policy=ppr")
                env_setup.append("")
        else:
            env_setup.append("# Using native launcher (ibrun/srun) - no OMPI_MCA variables needed")
            env_setup.append("")

        
        # Set environment variables
        if job_config.environment_variables:
            env_setup.append("# Set environment variables")
            for key, value in job_config.environment_variables.items():
                env_setup.append(f"export {key}={value}")
            env_setup.append("")
        
        return env_setup
    
    def _build_execution_section(self,
            job_config: SLURMJobConfig,
            executable_path: str,
            arguments: str,
            profiling: bool,
            likwid_enabled: bool,
            likwid_path: Optional[str] = None,
            enable_mpip: bool = True,
            mpip_path: Optional[str] = None,
            light_arguments: Optional[str] = None,
            multi_rank_profile: bool = False,
            nsight_duration_sec: Optional[int] = None,
            nsight_trace_domains: Optional[str] = None,
            spread_nsight_ranks: bool = False,
            likwid_rank0_only: bool = False) -> list:
        """Build job execution section for MPI"""
        execution = [
            "# Job execution",
            "# Stay in #SBATCH --chdir (scratch); EXPERIMENT_DIR holds artifacts (also on scratch when configured).",
            "echo \"Starting job: $SLURM_JOB_NAME\"",
            "echo \"Running on: $(hostname)\"",
            "echo \"MPI ranks per node: $SLURM_NTASKS_PER_NODE\"",
            "echo \"OpenMP threads per rank: $SLURM_CPUS_PER_TASK\"",
            "echo \"Total cores: $SLURM_NTASKS_PER_NODE * $SLURM_CPUS_PER_TASK\"",
            "echo \"Start time: $(date)\"",
            ""
        ]
        
        # Heavy per-job artifacts on scratch when hpc_config sets artifact_dir (LEAP2 GPFS-safe).
        experiment_dir_expr = experiment_artifact_dir_shell_expr(
            self.work_directory, self.system_config
        )
        
        # Profiling setup
        if profiling:
            profiling_setup = [
                "# Setup profiling",
                "export NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                "export NSYS_PROFILING_ENABLED=1",
                f"EXPERIMENT_DIR=\"{experiment_dir_expr}\"",
                "mkdir -p \"$EXPERIMENT_DIR\" || { echo \"FATAL: cannot create EXPERIMENT_DIR=$EXPERIMENT_DIR\" >&2; exit 1; }",
                "PROFILE_DIR=\"$EXPERIMENT_DIR/profiling/$SLURM_JOB_ID\"",
                "mkdir -p \"$PROFILE_DIR\"",
                "export EXPERIMENT_DIR PROFILE_DIR",
            ]
            if nsight_duration_sec is not None and nsight_duration_sec > 0:
                profiling_setup.append(f"export NSIGHT_DURATION={nsight_duration_sec}")
                profiling_setup.append("# Limit Nsight trace duration (faster profiling, α/β from slice)")
            profiling_setup.append("")
            execution.extend(profiling_setup)
        elif likwid_enabled:
            # Need to set EXPERIMENT_DIR even without profiling for LIKWID
            execution.extend([
                f"EXPERIMENT_DIR=\"{experiment_dir_expr}\"",
                "mkdir -p \"$EXPERIMENT_DIR\" || { echo \"FATAL: cannot create EXPERIMENT_DIR=$EXPERIMENT_DIR\" >&2; exit 1; }",
                "export EXPERIMENT_DIR",
            ])
            
        # LIKWID setup
        if likwid_enabled:
            likwid_lines = [
                "# LIKWID setup - find likwid-perfctr on compute node (multiple locations)",
            ]
            if likwid_path and likwid_path.strip():
                # User-provided path (e.g. from --likwid-path); prefer $HOME-relative on compute nodes
                likwid_lines.append(
                    f"export LIKWID_PATH={path_to_shell_home_literal(likwid_path.strip())}"
                )
            else:
                # Default: try $HOME/software/likwid so compute nodes find it when home is shared
                likwid_lines.append("[ -n \"$HOME\" ] && [ -d \"$HOME/software/likwid\" ] && export LIKWID_PATH=\"$HOME/software/likwid\"")
            likwid_lines.extend([
                "LIKWID_BIN=\"\"",
                "module load likwid 2>/dev/null || true",
                "if [ -n \"$LIKWID_PATH\" ] && [ -x \"$LIKWID_PATH/bin/likwid-perfctr\" ]; then",
                "    LIKWID_BIN=\"$LIKWID_PATH/bin\"",
                "elif [ -n \"$LIKWID_PATH\" ] && [ -x \"$LIKWID_PATH/likwid-perfctr\" ]; then",
                "    LIKWID_BIN=\"$LIKWID_PATH\"",
                "elif command -v likwid-perfctr >/dev/null 2>&1; then",
                "    LIKWID_BIN=\"$(dirname \"$(command -v likwid-perfctr)\")\"",
                "else",
                "    module load likwid 2>/dev/null || true",
                "    if command -v likwid-perfctr >/dev/null 2>&1; then",
                "        LIKWID_BIN=\"$(dirname \"$(command -v likwid-perfctr)\")\"",
                "    fi",
                "fi",
                "if [ -n \"$LIKWID_BIN\" ]; then",
                "    export PATH=\"$LIKWID_BIN:$PATH\"",
                "    export LIKWID_BIN",
                "    # Point LIKWID to its own perfgroup definitions to suppress ICX/system-path warnings.",
                "    _LIKWID_ROOT=\"$(dirname \"$LIKWID_BIN\")\"",
                "    _LIKWID_SHARE=\"\"",
                "    for _p in \"$_LIKWID_ROOT/share/likwid/perfgroups\" \"$_LIKWID_ROOT/../share/likwid/perfgroups\" \"$_LIKWID_ROOT/perfgroups\" \"$_LIKWID_ROOT/../perfgroups\"; do",
                "        if [ -d \"$_p\" ]; then _LIKWID_SHARE=\"$_p\"; break; fi",
                "    done",
                "    [ -n \"$_LIKWID_SHARE\" ] && export LIKWID_PERFGROUP_DIR=\"$_LIKWID_SHARE\"",
                "    if [ -n \"$LIKWID_PERFGROUP_DIR\" ]; then",
                "        echo \"LIKWID_PERFGROUP_DIR set to: $LIKWID_PERFGROUP_DIR\"",
                "    else",
                "        echo \"LIKWID_PERFGROUP_DIR: not set (tried: $_LIKWID_ROOT/{share/likwid/perfgroups,perfgroups} and ../variants)\"",
                "    fi",
                "fi",
                "export LIKWID_OUTPUT_DIR=\"$EXPERIMENT_DIR/likwid_output/$SLURM_JOB_ID\"",
                "mkdir -p $LIKWID_OUTPUT_DIR",
                "# Verify LIKWID is available",
                "if command -v likwid-perfctr >/dev/null 2>&1; then",
                "    echo \"LIKWID found: $(which likwid-perfctr)\"",
                "    # Cache -a for debugging; suppress cosmetic 'Cannot access .../ICX' path warnings.",
                "    LIKWID_GROUP_LIST=$(likwid-perfctr -a 2>&1 | grep -v 'Cannot access directory' || true)",
                "    export LIKWID_GROUP_LIST",
                "    # Probe which -g names actually run (tiny workload). Fixes \"No LIKWID locality group\" when -a is table/ANSI/unparsable.",
                "    LIKWID_RESOLVED_LOCALITY_GROUP=\"\"",
                "    LIKWID_RESOLVED_MEM_GROUP=\"\"",
                "    for _g in NUMA NDA L3 L3CACHE L2CACHE L2 MEMORY CBOX HBM; do",
                "        if likwid-perfctr -g \"$_g\" -c 0 -o \"$LIKWID_OUTPUT_DIR/.probe_l.csv\" -O -- true 2>/dev/null; then",
                "            LIKWID_RESOLVED_LOCALITY_GROUP=\"$_g\"",
                "            break",
                "        fi",
                "    done",
                "    rm -f \"$LIKWID_OUTPUT_DIR/.probe_l.csv\" 2>/dev/null || true",
                "    for _g in MEM MEM_DP DRAM MEMORY HBM L3CACHE; do",
                "        # MBOX/DRAM uncore counters need LIKWID_ACCESSMODE=perf_event (or accessdaemon/direct).",
                "        # Try with LIKWID_ACCESSMODE=perf_event so the probe succeeds on systems where",
                "        # /proc/sys/kernel/perf_event_paranoid <= 2 allows uncore access.",
                "        if LIKWID_ACCESSMODE=perf_event likwid-perfctr -g \"$_g\" -c 0 -o \"$LIKWID_OUTPUT_DIR/.probe_m.csv\" -O -- true 2>/dev/null; then",
                "            LIKWID_RESOLVED_MEM_GROUP=\"$_g\"",
                "            break",
                "        elif likwid-perfctr -g \"$_g\" -c 0 -o \"$LIKWID_OUTPUT_DIR/.probe_m.csv\" -O -- true 2>/dev/null; then",
                "            LIKWID_RESOLVED_MEM_GROUP=\"$_g\"",
                "            break",
                "        fi",
                "    done",
                "    rm -f \"$LIKWID_OUTPUT_DIR/.probe_m.csv\" 2>/dev/null || true",
                "    if { [ -z \"$LIKWID_RESOLVED_LOCALITY_GROUP\" ] || [ -z \"$LIKWID_RESOLVED_MEM_GROUP\" ]; } && [ -z \"${LIKWID_ACCESSMODE:-}\" ]; then",
                "        export LIKWID_ACCESSMODE=perf_event",
                "        echo \"LIKWID: first probe empty; retrying with LIKWID_ACCESSMODE=perf_event\"",
                "        for _g in NUMA NDA L3 L3CACHE L2CACHE L2 MEMORY CBOX HBM; do",
                "            if likwid-perfctr -g \"$_g\" -c 0 -o \"$LIKWID_OUTPUT_DIR/.probe_l.csv\" -O -- true 2>/dev/null; then",
                "                LIKWID_RESOLVED_LOCALITY_GROUP=\"$_g\"",
                "                break",
                "            fi",
                "        done",
                "        rm -f \"$LIKWID_OUTPUT_DIR/.probe_l.csv\" 2>/dev/null || true",
                "        for _g in MEM MEM_DP DRAM MEMORY HBM L3CACHE; do",
                "            if likwid-perfctr -g \"$_g\" -c 0 -o \"$LIKWID_OUTPUT_DIR/.probe_m.csv\" -O -- true 2>/dev/null; then",
                "                LIKWID_RESOLVED_MEM_GROUP=\"$_g\"",
                "                break",
                "            fi",
                "        done",
                "        rm -f \"$LIKWID_OUTPUT_DIR/.probe_m.csv\" 2>/dev/null || true",
                "    fi",
                "    export LIKWID_RESOLVED_LOCALITY_GROUP LIKWID_RESOLVED_MEM_GROUP",
                "    [ -n \"$LIKWID_RESOLVED_LOCALITY_GROUP\" ] && echo \"LIKWID: Run 1b group (probe)=$LIKWID_RESOLVED_LOCALITY_GROUP\"",
                "    [ -n \"$LIKWID_RESOLVED_MEM_GROUP\" ] && echo \"LIKWID: Run 1c group (probe)=$LIKWID_RESOLVED_MEM_GROUP\"",
                "    if [ -z \"$LIKWID_RESOLVED_LOCALITY_GROUP\" ] || [ -z \"$LIKWID_RESOLVED_MEM_GROUP\" ]; then",
                "        echo \"LIKWID: probe still missing locality or MEM — set LIKWID_ACCESSMODE (perf_event|accessdaemon|direct) or start likwid-accessD; check dmesg/counter access on compute nodes.\"",
                "    fi",
                "else",
                "    echo \"WARNING: LIKWID not found - locality metrics will be unavailable\"",
                "    echo \"  Tried: LIKWID_PATH, \\$HOME/software/likwid/bin, \\$SLURM_SUBMIT_DIR/software/likwid/bin, module load likwid\"",
                "    export LIKWID_GROUP_LIST=\"\"",
                "    export LIKWID_RESOLVED_LOCALITY_GROUP=\"\" LIKWID_RESOLVED_MEM_GROUP=\"\"",
                "fi",
                ""
            ])
            execution.extend(likwid_lines)
        
        # MPI execution
        mpi_exec = self._build_mpi_execution(
            job_config, executable_path, arguments, profiling, likwid_enabled, enable_mpip, mpip_path,
            light_arguments, multi_rank_profile, nsight_duration_sec, nsight_trace_domains, spread_nsight_ranks, likwid_rank0_only,
        )
        execution.extend(mpi_exec)
        
        return execution
    
    def _build_mpi_execution(self,
            job_config: SLURMJobConfig,
            executable_path: str,
            arguments: str,
            profiling: bool,
            likwid_enabled: bool,
            enable_mpip: bool = True,
            mpip_path: Optional[str] = None,
            light_arguments: Optional[str] = None,
            multi_rank_profile: bool = False,
            nsight_duration_sec: Optional[int] = None,
            nsight_trace_domains: Optional[str] = None,
            spread_nsight_ranks: bool = False,
            likwid_rank0_only: bool = False) -> list:
        """Build MPI execution section with proper multi-rank profiling support"""
        # Use mpirun (OpenMPI) instead of srun to avoid PMI incompatibility.
        # srun with OpenMPI requires SLURM built with --with-pmix, which is not always available.
        # mpirun (OpenMPI) works out-of-the-box and correctly inherits the environment.
        launcher_raw = job_config.additional_options.get("launcher", "mpirun")

        # Build the launcher invocation with correct flags for the detected launcher.
        # - mpirun (OpenMPI): uses -n, --map-by ppr:N:node:pe=M, inherits env by default.
        # - srun (SLURM native): uses --ntasks, --cpus-per-task, --export=ALL.
        if "mpirun" in launcher_raw or "orterun" in launcher_raw:
            # OpenMPI native launcher: map N ranks per node, each with M CPUs.
            ntasks_expr = "$SLURM_NTASKS"
            ntasks_node_expr = "$SLURM_NTASKS_PER_NODE"
            pe_map = getattr(job_config, "launcher_pe_per_rank", None)
            if pe_map is not None and str(pe_map).strip():
                pe_expr = str(pe_map).strip()
            else:
                pe_expr = "$SLURM_CPUS_PER_TASK"

            if _job_uses_mvapich2(job_config):
                # MVAPICH2: Hydra + -np/-ppn; OpenMPI --map-by / --bind-to are invalid here.
                fork = "-launcher fork " if job_config.nodes == 1 else ""
                launcher_cmd = (
                    f"{launcher_raw} {fork}-np {ntasks_expr} -ppn {ntasks_node_expr}"
                ).strip()
            else:
                launcher_cmd = (
                    f"{launcher_raw} -n {ntasks_expr} "
                    f"--map-by ppr:{ntasks_node_expr}:node:pe={pe_expr} "
                    f"--bind-to core"
                ).strip()
        elif "ibrun" == launcher_raw.strip():
            # TACC native launcher: ibrun reads everything from SLURM environment
            launcher_cmd = "ibrun"
        else:
            # srun / fallback: keep original srun-style flags
            export_flags = "--export=ALL"
            launcher_cmd = (
                f"{launcher_raw} --ntasks $SLURM_NTASKS "
                f"--cpus-per-task $SLURM_CPUS_PER_TASK {export_flags}"
            )

        # OpenMPI mpirun: Run 1a can forward mpiP via -x LD_PRELOAD (MVAPICH2/Hydra must not preload).
        _openmpi_mpip_preload = (
            enable_mpip
            and not _job_uses_mvapich2(job_config)
            and ("mpirun" in launcher_raw.lower() or "orterun" in launcher_raw.lower())
        )
        # MVAPICH2: Run 1a uses `srun` + LD_PRELOAD so only app ranks load mpiP (Hydra/mpirun + PMPI breaks).
        _mvapich_mpip_srun = (
            enable_mpip
            and _job_uses_mvapich2(job_config)
            and _coerce_additional_bool(job_config.additional_options.get("mpip_mvapich_srun"), True)
            and ("mpirun" in launcher_raw.lower() or "orterun" in launcher_raw.lower())
        )
        _srun_mpi_flag = (job_config.additional_options.get("srun_mpi_flag") or "").strip()
        if _srun_mpi_flag:
            _srun_mpi_flag_sh = _srun_mpi_flag.replace("'", "'\"'\"'")
        else:
            _srun_mpi_flag_sh = ""

        # Default Nsight: all ranks (Run 2 wraps full mpirun). Domains: OSRT + MPI + CUDA + cuBLAS.
        # mpiP: never Run 2 (Nsight / CUPTI). MVAPICH2: do not LD_PRELOAD mpiP on mpirun (Hydra + PMPI break). LIKWID: no mpiP in bash -c.
        # Optional --nsight-trace overrides (e.g. cuda,cublas for cheaper runs).
        _default_trace = "osrt,mpi,cuda,cublas,nvtx"
        if nsight_trace_domains and str(nsight_trace_domains).strip():
            _raw = str(nsight_trace_domains).strip()
            if re.fullmatch(r"[a-zA-Z0-9_,]+", _raw):
                trace_types = _raw
            else:
                trace_types = _default_trace
                logger.warning("Invalid nsight_trace_domains %r; using %s", _raw, _default_trace)
        else:
            trace_types = _default_trace
        
        # Only limit trace duration when explicitly set (no default). Full trace gives correct α/β/γ.
        nsys_duration_opts = ""
        if nsight_duration_sec is not None and nsight_duration_sec > 0:
            nsys_duration_opts = f" --duration={nsight_duration_sec}"
        
        # Define core commands
        core_cmd = f"{executable_path} {arguments}"
        light_core_cmd = f"{executable_path} {light_arguments.strip()}" if light_arguments and light_arguments.strip() else core_cmd
        
        mpi_exec = [
            "# Execute with MPI+OpenMP",
            "",
            "# ── Locate Nsight Systems executable (call resolve_nsys again before Run 2 if PATH changed) ──",
            "resolve_nsys() {",
            "    NSYS_EXE=\"\"",
            "    if command -v nsys >/dev/null 2>&1; then",
            "        NSYS_EXE=\"$(command -v nsys)\"",
            "    fi",
            "    if [ -z \"$NSYS_EXE\" ] && [ -n \"$PATH\" ]; then",
            "        _oifs=\"$IFS\"; IFS=':'",
            "        for _d in $PATH; do",
            "            if [ -n \"$_d\" ] && [ -x \"$_d/nsys\" ]; then NSYS_EXE=\"$_d/nsys\"; break; fi",
            "        done",
            "        IFS=\"$_oifs\"",
            "    fi",
            "    if [ -z \"$NSYS_EXE\" ] && [ -n \"$NSYS_HOME\" ] && [ -x \"$NSYS_HOME/bin/nsys\" ]; then",
            "        NSYS_EXE=\"$NSYS_HOME/bin/nsys\"",
            "    fi",
            "    if [ -z \"$NSYS_EXE\" ] && [ -n \"$CUDA_HOME\" ] && [ -x \"$CUDA_HOME/bin/nsys\" ]; then",
            "        NSYS_EXE=\"$CUDA_HOME/bin/nsys\"",
            "    fi",
            "    export NSYS_EXE",
            "}",
            "resolve_nsys",
            "if [ -n \"$NSYS_EXE\" ]; then",
            "    echo \"Nsight: using $NSYS_EXE\"",
            "    \"$NSYS_EXE\" --version 2>/dev/null | head -1 || true",
            "else",
            "    echo \"WARNING: nsys not found (PATH/NSYS_HOME/CUDA_HOME) — Run 2 will be skipped; check --env-path or module nsight-systems/cuda\"",
            "    echo \"PATH (first 20 entries):\"; echo \"$PATH\" | tr ':' '\\012' | head -20",
            "fi",
            "",
        ]
        # mpiP: OpenMPI Run 1a may use -x LD_PRELOAD; MVAPICH2 Run 1a may use srun+LD_PRELOAD. Run 2 never preloads mpiP.
        if enable_mpip:
            mpip_candidates = []
            if mpip_path and mpip_path.strip():
                _mpip_lit = path_to_shell_home_literal(mpip_path.strip())
                if _mpip_lit:
                    mpip_candidates.append(f"export MPIP_PATH={_mpip_lit}")
                mpip_candidates.append("")
                mpip_candidates.append("for _CANDIDATE in \\")
                mpip_candidates.append("    \"$MPIP_PATH/lib/libmpiP.so\" \\")
            else:
                mpip_candidates.append("for _CANDIDATE in \\")
            mpip_candidates.extend([
                "    \"$HOME/software/mpiP/lib/libmpiP.so\" \\",
                "    \"$(pwd)/mpiP/lib/libmpiP.so\" \\",
                "    \"/usr/local/lib/libmpiP.so\"; do",
                "    if [ -f \"$_CANDIDATE\" ]; then",
                "        MPIP_LIB=\"$_CANDIDATE\"",
                "        break",
                "    fi",
                "done",
                "",
                "if [ -n \"$MPIP_LIB\" ]; then",
                "    MPIP_OUT_DIR=\"$EXPERIMENT_DIR/mpiP/${SLURM_JOB_ID}\"",
                "    mkdir -p \"$MPIP_OUT_DIR\"",
                "    mkdir -p \"$MPIP_OUT_DIR/report\"",
                "    # mpiP -f writes to <dir>/report/<name>.<nprocs>.<pid>.<rank>.mpiP; .err showed 'Could not open .../report/...' without this.",
                "    # Do NOT export LD_PRELOAD here: only Run 1a subshell / OpenMPI -x injects preload.",
                "    # MVAPICH2: mpirun + LD_PRELOAD breaks Hydra; Run 1a uses srun when MPIP_MVAPICH_SRUN=1.",
                "    export MPIP_LIB MPIP_OUT_DIR",
                "    export MPIP=\"-f $MPIP_OUT_DIR/report -k 0\"",
                "    if [ \"${MPIP_RUN1_PRELOAD:-0}\" = \"1\" ]; then",
                "        echo \"mpiP: MPIP_LIB=$MPIP_LIB (OpenMPI: Run 1a uses -x LD_PRELOAD; reports under $MPIP_OUT_DIR/report)\"",
                "    elif [ \"${MPIP_MVAPICH_SRUN:-0}\" = \"1\" ]; then",
                "        echo \"mpiP: MPIP_LIB=$MPIP_LIB (MVAPICH2: Run 1a uses srun+LD_PRELOAD; reports under $MPIP_OUT_DIR/report)\"",
                "    else",
                "        echo \"mpiP: MPIP_LIB=$MPIP_LIB (no Run-1a preload — α from Nsight if MVAPICH2 without srun path)\"",
                "    fi",
                "fi",
                "",
            ])
            mpi_exec.append("# ── mpiP: α (communication) from PMPI-level timing ─────────")
            mpi_exec.append("MPIP_LIB=\"\"")
            mpi_exec.append(f"export MPIP_RUN1_PRELOAD={'1' if _openmpi_mpip_preload else '0'}")
            mpi_exec.append(f"export MPIP_MVAPICH_SRUN={'1' if _mvapich_mpip_srun else '0'}")
            mpi_exec.append(f"export SRUN_MPI_FLAG='{_srun_mpi_flag_sh}'")
            mpi_exec.extend(mpip_candidates)
        else:
            mpi_exec.append("# mpiP disabled (--no-mpip)")
            mpi_exec.append("export MPIP_RUN1_PRELOAD=0")
            mpi_exec.append("export MPIP_MVAPICH_SRUN=0")
            mpi_exec.append("export SRUN_MPI_FLAG=\"\"")
            mpi_exec.append("")
        mpi_exec.append(
            "# ε (OpenMP): derived post-hoc from Nsight osrt/scheduling + CPU util (see metrics_extractor)."
        )
        
        # ---------------------------------------------------------
        # Unified Execution Logic
        # ---------------------------------------------------------
        
        # Run 1a: Performance Application (Standard) - ALWAYS full arguments for genuine runtime/throughput.
        # Use core_cmd (full user args) so "Run 1a Time:" and throughput are from the real workload.
        run1_cmd = core_cmd

        # MVAPICH2 + mpiP: Run 1a uses srun + LD_PRELOAD (only MPI tasks; no Hydra preload).
        # OpenMPI: Run 1a uses RUN1_LAUNCH with -x LD_PRELOAD=$MPIP_LIB when MPIP_RUN1_PRELOAD=1.
        if _openmpi_mpip_preload:
            _run1_launch_setup = [
                "if [ \"$MPIP_RUN1_PRELOAD\" = \"1\" ] && [ -n \"$MPIP_LIB\" ]; then",
                f"    RUN1_LAUNCH=\"{launcher_cmd} -x LD_PRELOAD=\\$MPIP_LIB\"",
                "else",
                f"    RUN1_LAUNCH=\"{launcher_cmd}\"",
                "fi",
            ]
        elif _mvapich_mpip_srun:
            _srun_cpu_bind = "--cpu-bind=cores"
            _run1_launch_setup = [
                "if [ \"${MPIP_MVAPICH_SRUN:-0}\" = \"1\" ] && [ -n \"$MPIP_LIB\" ]; then",
                "    RUN1_LAUNCH=\"srun --ntasks $SLURM_NTASKS --cpus-per-task $SLURM_CPUS_PER_TASK "
                f"--export=ALL,LD_PRELOAD,MPIP {_srun_cpu_bind} $SRUN_MPI_FLAG\"",
                "else",
                f"    RUN1_LAUNCH=\"{launcher_cmd}\"",
                "fi",
            ]
        else:
            _run1_launch_setup = [f"RUN1_LAUNCH=\"{launcher_cmd}\""]

        # OpenMPI: unset LD_PRELOAD before mpirun; it re-injects via -x. MVAPICH srun: set LD_PRELOAD in subshell for srun --export.
        run1a_line = (
            "( if [ \"${MPIP_MVAPICH_SRUN:-0}\" = \"1\" ] && [ -n \"$MPIP_LIB\" ]; then "
            "export LD_PRELOAD=\"$MPIP_LIB\"; else unset LD_PRELOAD; fi; "
            "if command -v stdbuf >/dev/null 2>&1; then stdbuf -oL -eL $RUN1_LAUNCH $RUN_CMD; "
            "else $RUN1_LAUNCH $RUN_CMD; fi 2>&1 || true ) | tee \"$RUN1_OUT_LOCAL\""
        )

        mpi_exec.extend([
            "# ---------------------------------------------------------",
            "# Run 1a: Performance Application (Standard)",
            "# ---------------------------------------------------------",
            "echo \"Starting Run 1a (Performance - Clean)...\"",
            f"RUN_CMD=\"{run1_cmd}\"",
            "export RUN_CMD",
            "",
            "RUN1_OUT_LOCAL=\"${TMPDIR:-/tmp}/run1_${SLURM_JOB_ID}.out\"",
            "echo \"Running standard application (Run 1a)...\"",
            "# Run 1a: OpenMPI unsets LD_PRELOAD then mpirun -x reinjects; MVAPICH srun path exports LD_PRELOAD for app tasks only.",
            "# stdbuf: line-buffered MPI/LULESH stdout so timing lines land in slurm-*.out before Run 1b.",
            *_run1_launch_setup,
            run1a_line,
            "",
            "# Create global Run 1 output file immediately",
            "cp -f \"$RUN1_OUT_LOCAL\" \"$EXPERIMENT_DIR/run1_${SLURM_JOB_ID}.out\" 2>/dev/null || true",
            "",
            "# Parse and print Runtime/Throughput immediately for logging",
            "# HYBRID_VEC_GPU/SPARSE: Time=... | LULESH: Elapsed time = X (s)",
            # grep/sed can exit non-zero when no match; with `set -e` that must not abort the batch step.
            "FULL_RUNTIME=$(grep -E 'Time=' \"$RUN1_OUT_LOCAL\" 2>/dev/null | tail -1 | sed -n 's/.*Time=\\([0-9.]*\\)[^0-9].*/\\1/p' || true)",
            "if [ -z \"$FULL_RUNTIME\" ]; then",
            # sed -E: + is “one or more” (BRE default treats + literally, so Run 1a Time was never echoed).
            "    FULL_RUNTIME=$(grep -E 'Elapsed[[:space:]]+time' \"$RUN1_OUT_LOCAL\" 2>/dev/null | tail -1 | sed -nE 's/.*Elapsed[[:space:]]+time[[:space:]]*=[[:space:]]*([0-9.eE+-]+).*/\\1/p' || true)",
            "fi",
            *slurm_full_runtime_fallback_from_run1_out_lines(),
            "if [ -n \"$FULL_RUNTIME\" ]; then",
            "    echo \"Run 1a Time: $FULL_RUNTIME seconds\"",
            "fi",
            "FULL_FOM=$(grep -E 'FOM[[:space:]]*=' \"$RUN1_OUT_LOCAL\" 2>/dev/null | tail -1 | sed -n 's/.*FOM[[:space:]]*=[[:space:]]*\\([0-9.eE+-]*\\).*/\\1/p' || true)",
            "if [ -n \"$FULL_FOM\" ]; then",
            "    echo \"Run 1a FOM: $FULL_FOM z/s\"",
            "fi",
            ""
        ])

        # Run 1b: LIKWID (Optional) - Separate Pass
        # LIKWID wraps each MPI rank with hardware counter collection (or rank 0 only when likwid_rank0_only).
        # Uses a per-rank wrapper script for reliable per-rank csv output.
        if likwid_enabled:
            if likwid_rank0_only:
                likwid_wrapper = (
                    f"{launcher_cmd} "
                    "bash -c '"
                    "MPI_RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${MV2_COMM_WORLD_RANK:-}}}; "
                    "[ -z \"$MPI_RANK\" ] && MPI_RANK=999; "
                    "if [ -n \"$LIKWID_BIN\" ]; then "
                    f"  C_MAX=$(( {job_config.cpus_per_task} - 1 )); "
                    "  G_LOCALITY=\"${LIKWID_RESOLVED_LOCALITY_GROUP:-}\"; "
                    "  if [ -z \"$G_LOCALITY\" ]; then "
                    "    _GL=\"${LIKWID_GROUP_LIST:-}\"; "
                    "    [ -z \"$_GL\" ] && [ -n \"$LIKWID_BIN\" ] && _GL=$(\"$LIKWID_BIN\"/likwid-perfctr -a 2>&1 || true); "
                    "    for _g in NUMA NDA L3 L3CACHE L2CACHE L2 MEMORY CBOX HBM; do "
                    "      printf '%s\\n' \"$_GL\" | grep -qiw -- \"$_g\" && { G_LOCALITY=$_g; break; }; "
                    "    done; "
                    "  fi; "
                    "  if [ -n \"$G_LOCALITY\" ] && [ \"$MPI_RANK\" -eq 0 ]; then "
                    # Fixed names: global rank 0 only; matches get_locality_for_job / likwid_0.csv
                    "    taskset -c 0-$C_MAX \"$LIKWID_BIN\"/likwid-perfctr -g \"$G_LOCALITY\" -c 0-$C_MAX -o \"$LIKWID_OUTPUT_DIR/likwid_0.csv\" "
                    f"    {run1_cmd} || echo \"WARNING: LIKWID locality run timed out or failed (ignoring)\"; "
                    "  elif [ -z \"$G_LOCALITY\" ] && [ \"$MPI_RANK\" -eq 0 ]; then "
                    "    echo \"WARNING: No LIKWID locality group (NUMA/L3/L2/…); running bare on rank 0\"; "
                    f"    {run1_cmd} || true; "
                    "  else "
                    f"    {run1_cmd} || true; "
                    "  fi; "
                    "else "
                    "  echo \"LIKWID not found, skipping 1b\"; "
                    "fi'"
                )
                likwid_wrapper_mem = (
                    f"{launcher_cmd} "
                    "bash -c '"
                    "MPI_RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${MV2_COMM_WORLD_RANK:-}}}; "
                    "[ -z \"$MPI_RANK\" ] && MPI_RANK=999; "
                    "if [ -n \"$LIKWID_BIN\" ]; then "
                    f"  C_MAX=$(( {job_config.cpus_per_task} - 1 )); "
                    "  G_MEM=\"${LIKWID_RESOLVED_MEM_GROUP:-}\"; "
                    "  if [ -z \"$G_MEM\" ]; then "
                    "    _GL=\"${LIKWID_GROUP_LIST:-}\"; "
                    "    [ -z \"$_GL\" ] && [ -n \"$LIKWID_BIN\" ] && _GL=$(\"$LIKWID_BIN\"/likwid-perfctr -a 2>&1 || true); "
                    "    for _g in MEM MEM_DP DRAM MEMORY HBM L3CACHE; do "
                    "      printf '%s\\n' \"$_GL\" | grep -qiw -- \"$_g\" && { G_MEM=$_g; break; }; "
                    "    done; "
                    "  fi; "
                    "  if [ -n \"$G_MEM\" ] && [ \"$MPI_RANK\" -eq 0 ]; then "
                "    LIKWID_ACCESSMODE=perf_event taskset -c 0-$C_MAX \"$LIKWID_BIN\"/likwid-perfctr -g \"$G_MEM\" -c 0-$C_MAX -o \"$LIKWID_OUTPUT_DIR/likwid_mem_0.csv\" "
                f"    {run1_cmd} || echo \"WARNING: LIKWID MEM run timed out or failed\"; "
                    "  elif [ -z \"$G_MEM\" ] && [ \"$MPI_RANK\" -eq 0 ]; then "
                    "    echo \"WARNING: No LIKWID MEM group (MEM/MEM_DP/…); running bare on rank 0\"; "
                    f"    {run1_cmd} || true; "
                    "  else "
                    f"    {run1_cmd} || true; "
                    "  fi; "
                    "else "
                    "  echo \"LIKWID not found, skipping 1c\"; "
                    "fi'"
                )
            else:
                likwid_wrapper = (
                    f"{launcher_cmd} "
                    "bash -c '"
                    "R=${SLURM_PROCID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${MV2_COMM_WORLD_RANK:-0}}}}; "
                    # Only rank 0 runs LIKWID to avoid flooding .out with N×cpuset warnings when there
                    # are many ranks (e.g. 96-rank pure-MPI job → 96 LIKWID instances × 1 CPU each).
                    # All other ranks run the application bare (they participate in the MPI job normally).
                    "if [ -n \"$LIKWID_BIN\" ] && [ \"$R\" -eq 0 ]; then "
                    f"  C_MAX=$(( {job_config.cpus_per_task} - 1 )); "
                    "  G_LOCALITY=\"${LIKWID_RESOLVED_LOCALITY_GROUP:-}\"; "
                    "  if [ -z \"$G_LOCALITY\" ]; then "
                    "    _GL=\"${LIKWID_GROUP_LIST:-}\"; "
                    "    [ -z \"$_GL\" ] && _GL=$(\"$LIKWID_BIN\"/likwid-perfctr -a 2>&1 || true); "
                    "    for _g in NUMA NDA L3 L3CACHE L2CACHE L2 MEMORY CBOX HBM; do "
                    "      printf '%s\\n' \"$_GL\" | grep -qiw -- \"$_g\" && { G_LOCALITY=$_g; break; }; "
                    "    done; "
                    "  fi; "
                    "  if [ -n \"$G_LOCALITY\" ]; then "
                    "    taskset -c 0-$C_MAX \"$LIKWID_BIN\"/likwid-perfctr -g \"$G_LOCALITY\" -c 0-$C_MAX -o \"$LIKWID_OUTPUT_DIR/likwid_$R.csv\" "
                    f"    {run1_cmd} 2>/dev/null || echo \"WARNING: LIKWID locality run timed out or failed (ignoring)\"; "
                    "  else "
                    "    echo \"WARNING: No LIKWID locality group (NUMA/L3/L2/…); running bare on rank $R\"; "
                    f"    {run1_cmd} || true; "
                    "  fi; "
                    "else "
                    f"  {run1_cmd} || true; "
                    "fi'"
                )
                likwid_wrapper_mem = (
                    f"{launcher_cmd} "
                    "bash -c '"
                    "R=${SLURM_PROCID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${MV2_COMM_WORLD_RANK:-0}}}}; "
                    "if [ -n \"$LIKWID_BIN\" ] && [ \"$R\" -eq 0 ]; then "
                    f"  C_MAX=$(( {job_config.cpus_per_task} - 1 )); "
                    "  G_MEM=\"${LIKWID_RESOLVED_MEM_GROUP:-}\"; "
                    "  if [ -z \"$G_MEM\" ]; then "
                    "    _GL=\"${LIKWID_GROUP_LIST:-}\"; "
                    "    [ -z \"$_GL\" ] && [ -n \"$LIKWID_BIN\" ] && _GL=$(\"$LIKWID_BIN\"/likwid-perfctr -a 2>&1 || true); "
                    "    for _g in MEM MEM_DP DRAM MEMORY HBM L3CACHE; do "
                    "      printf '%s\\n' \"$_GL\" | grep -qiw -- \"$_g\" && { G_MEM=$_g; break; }; "
                    "    done; "
                    "  fi; "
                    "  if [ -n \"$G_MEM\" ]; then "
                    "    LIKWID_ACCESSMODE=perf_event taskset -c 0-$C_MAX \"$LIKWID_BIN\"/likwid-perfctr -g \"$G_MEM\" -c 0-$C_MAX -o \"$LIKWID_OUTPUT_DIR/likwid_mem_$R.csv\" "
                    f"    {run1_cmd} 2>/dev/null || echo \"WARNING: LIKWID MEM run timed out or failed\"; "
                    "  else "
                    "    echo \"WARNING: No LIKWID MEM group (MEM/MEM_DP/…); running bare on rank $R\"; "
                    f"    {run1_cmd} || true; "
                    "  fi; "
                    "else "
                    f"  {run1_cmd} || true; "
                    "fi'"
                )
            # Outer mpirun can exit non-zero (MPI abort, Hydra, CUDA MPS) while inner rank shells used
            # `|| true`. Under `set -e` the batch step fails → SLURM FAILED + empty results dir (no slurm-*.out).
            _likwid_r1b_guard = (
                " || { echo \"WARNING: Run 1b LIKWID locality launcher failed (exit $?); continuing job.\" >&2; true; }"
            )
            _likwid_r1c_guard = (
                " || { echo \"WARNING: Run 1c LIKWID MEM launcher failed (exit $?); continuing job.\" >&2; true; }"
            )
            mpi_exec.extend([
                "# ---------------------------------------------------------",
                "# Run 1b: LIKWID Locality (Group: NUMA)",
                "# mpiP: Run 1a only (OpenMPI -x / MVAPICH srun). Do not LD_PRELOAD mpiP inside this bash.",
                "# ---------------------------------------------------------",
                "echo \"Starting Run 1b (LIKWID Locality)...\"",
                # Snapshot kernel NUMA page-allocation counters before Run 1b.
                # Two separate sysfs files may carry NUMA allocation stats (readable by any user):
                #   1. /sys/devices/system/node/nodeN/numastat  — always present on NUMA kernels
                #      Fields: numa_hit, numa_miss, local_node, other_node (lowercase, no 'Node N' prefix)
                #   2. /sys/devices/system/node/nodeN/meminfo   — only has NUMA fields when CONFIG_NUMA_STAT=y
                #      Fields: Node N LocalNode:  NNNNN  /  Node N OtherNode:  NNNNN
                # We try numastat first, fall back to meminfo. Diff before/after → delta for this run only.
                "_NUMA_STAT_BEFORE=''",
                "_NUMA_STAT_SRC=''",
                "if [ -d /sys/devices/system/node/node0 ]; then",
                "    # Try numastat first (always present on NUMA-capable kernels)",
                "    # Use local_node / other_node only (process-perspective NUMA locality).",
                "    # Both fields are present on every NUMA-capable Linux kernel.",
                "    _NUMA_STAT_BEFORE=$(cat /sys/devices/system/node/node*/numastat 2>/dev/null | grep -iE '^(local_node|other_node)' || true)",
                "    if [ -n \"$_NUMA_STAT_BEFORE\" ]; then",
                "        _NUMA_STAT_SRC='numastat'",
                "        _NUMA_LINE_CT=$(echo \"$_NUMA_STAT_BEFORE\" | wc -l)",
                "        echo \"NUMA sysfs: using numastat — captured ${_NUMA_LINE_CT} counter lines before run\"",
                "    else",
                "        # Fall back to meminfo NUMA fields (CONFIG_NUMA_STAT kernels)",
                "        _NUMA_STAT_BEFORE=$(cat /sys/devices/system/node/node*/meminfo 2>/dev/null | grep -iE 'LocalNode|OtherNode|NumaHit|NumaMiss' || true)",
                "        if [ -n \"$_NUMA_STAT_BEFORE\" ]; then",
                "            _NUMA_STAT_SRC='meminfo'",
                "            _NUMA_LINE_CT=$(echo \"$_NUMA_STAT_BEFORE\" | wc -l)",
                "            echo \"NUMA sysfs: using meminfo NUMA fields — captured ${_NUMA_LINE_CT} lines before run\"",
                "        else",
                "            echo \"NUMA sysfs: no NUMA allocation counters available (numastat and meminfo both empty)\"",
                "            echo \"  numastat head:\"; head -4 /sys/devices/system/node/node0/numastat 2>/dev/null || echo '  (not found)'",
                "            echo \"  meminfo head:\"; head -8 /sys/devices/system/node/node0/meminfo 2>/dev/null || echo '  (not found)'",
                "        fi",
                "    fi",
                "else",
                "    echo \"NUMA sysfs: /sys/devices/system/node/ not found on this compute node — skipping snapshot\"",
                "fi",
            ])
            if likwid_rank0_only:
                mpi_exec.append(
                    "echo \"LIKWID Run 1b/1c: rank 0 only (OMPI_COMM_WORLD_RANK/PMI_RANK); other ranks run bare.\""
                )
            mpi_exec.extend([
                likwid_wrapper + _likwid_r1b_guard,
                "sleep 2",
                "sync",
                # Post-run kernel NUMA snapshot.
                # Write BEFORE/AFTER/SOURCE to numa_meminfo.txt for the Python parser.
                "if [ -n \"$_NUMA_STAT_SRC\" ] && [ -d /sys/devices/system/node/node0 ]; then",
                "    if [ \"$_NUMA_STAT_SRC\" = 'numastat' ]; then",
                "        _NUMA_STAT_AFTER=$(cat /sys/devices/system/node/node*/numastat 2>/dev/null | grep -iE '^(local_node|other_node)' || true)",
                "    else",
                "        _NUMA_STAT_AFTER=$(cat /sys/devices/system/node/node*/meminfo 2>/dev/null | grep -iE 'LocalNode|OtherNode|NumaHit|NumaMiss' || true)",
                "    fi",
                "    if [ -n \"$_NUMA_STAT_BEFORE\" ] && [ -n \"$_NUMA_STAT_AFTER\" ]; then",
                "        printf 'SOURCE\\n%s\\nBEFORE\\n%s\\nAFTER\\n%s\\n' \"$_NUMA_STAT_SRC\" \"$_NUMA_STAT_BEFORE\" \"$_NUMA_STAT_AFTER\" > \"$LIKWID_OUTPUT_DIR/numa_meminfo.txt\"",
                "        echo \"NUMA sysfs snapshot saved (src=$_NUMA_STAT_SRC): $LIKWID_OUTPUT_DIR/numa_meminfo.txt\"",
                "    fi",
                "fi",
                ""
            ])
            
            # Run 1c: LIKWID Bandwidth (Group: MEM)
            # We run this separately because LIKWID usually cannot measure NUMA and MEM groups simultaneously.
            mpi_exec.extend([
                "# ---------------------------------------------------------",
                "# Run 1c: LIKWID Bandwidth (Group: MEM)",
                "# ---------------------------------------------------------",
                "echo \"Starting Run 1c (LIKWID Bandwidth)...\"",
                likwid_wrapper_mem + _likwid_r1c_guard,
                "sleep 2",
                "sync",
                # Post-run MBOX zero-check (outside bash -c to avoid single-quote nesting issues).
                # Reports a clear diagnostic if DRAM counters are all zero (needs perf_event_paranoid<=2).
                # NOTE: grep -cE exits code 1 when 0 matches are found (but still prints "0" to stdout).
                # Using || true (not || echo 0) prevents a duplicate "0" being appended which would make
                # the variable "0\n0" and cause [: 0\n0: integer expression expected.
                "if [ -n \"$LIKWID_BIN\" ] && [ -f \"$LIKWID_OUTPUT_DIR/likwid_mem_0.csv\" ]; then",
                "    _mbox_l=$(grep -cE 'CAS_COUNT|MBOX' \"$LIKWID_OUTPUT_DIR/likwid_mem_0.csv\" 2>/dev/null || true)",
                "    _mbox_nz=$(grep -E 'CAS_COUNT|MBOX' \"$LIKWID_OUTPUT_DIR/likwid_mem_0.csv\" 2>/dev/null | grep -cvE ',0(,|$)' || true)",
                "    if [ \"${_mbox_l:-0}\" -gt 0 ] && [ \"${_mbox_nz:-0}\" -eq 0 ]; then",
                "        echo \"WARNING: LIKWID MEM/MBOX counters are all zero — DRAM BW unavailable.\"",
                "        echo \"  Fix: ask admin to lower /proc/sys/kernel/perf_event_paranoid, or start likwid-accessD.\"",
                "    fi",
                "fi",
                ""
            ])

        
        # For MVAPICH2 (single or multi-rank): strip the 'mpi' trace domain.
        # nsys 2026.x PMPI intercept layer for MVAPICH2 causes a segfault on cleanup regardless
        # of rank count; removing 'mpi' fixes both the multi-rank crash and the persistent single-rank
        # segfault seen for 1x48 (large thread count amplifies teardown race in Nsight's MPI wrapper).
        # CPU/OSRT/scheduling metrics (β, ε) still come from the remaining trace domains.
        # α is covered by mpiP from Run 1a so this is not a metric loss.
        _effective_trace = trace_types
        if _job_uses_mvapich2(job_config):
            _parts = [t.strip() for t in trace_types.split(",") if t.strip().lower() != "mpi"]
            _effective_trace = ",".join(_parts) if _parts else "osrt"
        # Alias kept for readability in the multi-rank command block below.
        _multi_rank_trace = _effective_trace

        nsys_run_cmd_rank0 = (
            f"$NSYS_EXE profile -o \"$PROFILE_DIR/profile\" --force-overwrite true "
            f"--trace={_effective_trace}{nsys_duration_opts} "
            "$PROFILE_CMD"
        )

        # Run 2: Profiling (Only if requested)
        if profiling:
            # Run 2 usually profiles the workload (light)
            run2_cmd = light_core_cmd
            
            mpi_exec.extend([
                "# ---------------------------------------------------------",
                "# Run 2: Nsight Systems Profiling",
                "# ---------------------------------------------------------",
                "resolve_nsys",
                "if [ -n \"$NSYS_EXE\" ]; then",
                "    echo \"Starting Run 2 (Profiling with Nsight)...\"",
                f"    echo \"Policy: all MPI ranks under Nsight (nsys wraps full launcher); --trace={_effective_trace}; mpiP unset (MVAPICH2/OpenMPI launch).\"",
                f"    PROFILE_CMD=\"{run2_cmd}\"",
                "    export PROFILE_CMD",
                ""
            ])

            # Multi-rank: nsys must wrap the full launcher (mpirun/srun + app), not only rank 0's app.
            # The old pattern (mpirun → per-rank shell: rank0 runs nsys profile app, others run app) made
            # rank 0's MPI program a child of nsys instead of orted, which led to MPI/CUDA hangs until
            # SLURM TIMEOUT despite a small workload. One nsys session tracing the whole job is correct.
            # spread_nsight_ranks: same unified trace (one profile.sqlite); per-rank .nsys-rep files removed.
            # Use `case` on SLURM_NTASKS (not `[ … -eq 1 ]`): mis-set or empty SLURM_NTASKS caused
            # "integer expression expected" under `set -e` → batch FAILED after Run 2 even when nsys ran.
            multi_rank_nsys_wrap_launcher = [
                "        *)",
                "        # Multi-rank: Nsight wraps the full MPI launch (all ranks are direct MPI app processes).",
                "        unset LD_PRELOAD 2>/dev/null || true",
                "        [ -n \"$MPIP_OUT_DIR\" ] && mkdir -p \"$MPIP_OUT_DIR/report\"",
                "        rm -f \"$PROFILE_DIR/nvidia_smi_util_samples.csv\"",
                "        NV_SAMP_PID=\"\"",
                "        if command -v nvidia-smi >/dev/null 2>&1; then",
                "            ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 >> \"$PROFILE_DIR/nvidia_smi_util_samples.csv\" || true; sleep 0.2; done ) &",
                "            NV_SAMP_PID=$!",
                "        fi",
                f"        $NSYS_EXE profile -o \"$PROFILE_DIR/profile\" --force-overwrite true "
                f"--trace={_multi_rank_trace}{nsys_duration_opts} {launcher_cmd} $PROFILE_CMD || true",
                "        if [ -n \"${NV_SAMP_PID:-}\" ]; then kill $NV_SAMP_PID 2>/dev/null || true; wait $NV_SAMP_PID 2>/dev/null || true; fi",
                "        ;;",
            ]
            export_rank0 = [
                "",
                "    # Export full-job .nsys-rep (all ranks in session) to SQLite for metrics extraction",
                "    FOUND_REP=\"$PROFILE_DIR/profile.nsys-rep\"",
                "    if [ ! -f \"$FOUND_REP\" ]; then",
                "        FOUND_REP=$(find \"$PROFILE_DIR\" -maxdepth 1 -name \"*.nsys-rep\" 2>/dev/null | head -n 1 || true)",
                "    fi",
                "    if [ -n \"$FOUND_REP\" ] && [ -f \"$FOUND_REP\" ]; then",
                "        echo \"Found profile trace: $FOUND_REP\"",
                "        echo \"Exporting profile to SQLite...\"",
                "        $NSYS_EXE export --type sqlite --output \"$PROFILE_DIR/profile.sqlite\" \"$FOUND_REP\" 2>&1 || {",
                "            echo \"WARNING: SQLite export failed, trying fallback...\"",
                "            $NSYS_EXE export --output \"$PROFILE_DIR/profile.sqlite\" \"$FOUND_REP\" 2>&1 || true",
                "        }",
                "    else",
                "        echo \"WARNING: No .nsys-rep file found in $PROFILE_DIR\"",
                "        ls -la \"$PROFILE_DIR\" 2>/dev/null || true",
                "    fi",
            ]
            mpi_exec.extend(
                [
                    "    case \"${SLURM_NTASKS:-1}\" in",
                    "        1)",
                    "        # Single Rank: Run nsys directly on the application",
                    "        # Hybrid: no mpiP on Run 2 (CUPTI); α from Nsight / OSRT-MPI tables.",
                    "        unset LD_PRELOAD 2>/dev/null || true",
                    "        [ -n \"$MPIP_OUT_DIR\" ] && mkdir -p \"$MPIP_OUT_DIR/report\"",
                    "        rm -f \"$PROFILE_DIR/nvidia_smi_util_samples.csv\"",
                    "        NV_SAMP_PID=\"\"",
                    "        if command -v nvidia-smi >/dev/null 2>&1; then",
                    "            ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 >> \"$PROFILE_DIR/nvidia_smi_util_samples.csv\" || true; sleep 0.2; done ) &",
                    "            NV_SAMP_PID=$!",
                    "        fi",
                    f"        {nsys_run_cmd_rank0} || true",
                    "        if [ -n \"${NV_SAMP_PID:-}\" ]; then kill $NV_SAMP_PID 2>/dev/null || true; wait $NV_SAMP_PID 2>/dev/null || true; fi",
                    "        ;;",
                ]
                + multi_rank_nsys_wrap_launcher
                + [
                    "    esac",
                ]
                + export_rank0
                + [
                    "else",
                    "    echo \"Nsight not found. Skipping Run 2.\"",
                    "fi",
                    "",
                ]
            )
            
        mpi_exec.append("")
        return mpi_exec


    
    def _build_cleanup_section(self, profiling: bool) -> list:
        """Build cleanup and reporting section"""
        cleanup = [
            "# Job completion",
            "echo \"End time: $(date)\"",
            "echo \"Job completed successfully\"",
            ""
        ]
        
        if profiling:
            cleanup.extend([
                "# Profiling cleanup",
                "if [ -d \"$PROFILE_DIR\" ]; then",
                "    echo \"Profiling data saved to: $PROFILE_DIR\"",
                "    ls -la $PROFILE_DIR",
                "fi"
            ])
        
        return cleanup
