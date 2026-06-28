#!/usr/bin/env python3
"""
SLURM Job Manager for HPC Systems

Main orchestrator that coordinates the modular SLURM management components.
Portable across any SLURM-based HPC system.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from .slurm.models import SLURMJobConfig, JobStatus
from .slurm.config import HPCSystemConfig
from autotuner.utils.path_utils import slurm_template_to_host_path
from .slurm.job_generator import SLURMJobGenerator
from .slurm.job_monitor import SLURMJobMonitor
from .slurm.result_collector import SLURMResultCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlurmManager:
    """
    Main SLURM manager that orchestrates all SLURM operations
    
    This is the primary interface for SLURM job management, coordinating
    between the specialized modules for configuration, generation, monitoring,
    and result collection.
    """
    
    def __init__(self, work_directory: Path, system_name: str = "default"):
        """
        Initialize the SLURM manager
        
        Args:
            work_directory: Working directory for jobs and results
            system_name: HPC system name (used for config lookup, or 'default')
        """
        self.work_directory = Path(work_directory)
        self.system_name = (system_name or "default").lower()
        
        # Create working directory structure
        self._setup_working_directory()
        
        # Load system-specific configuration
        self.system_config = HPCSystemConfig.get_system_config(self.system_name)
        
        # Validate and handle QOS - disable if not available
        self._validate_qos()
        self._ensure_scratch_slurm_dirs()

        # Initialize component managers
        self.job_generator = SLURMJobGenerator(
            self.work_directory, system_config=self.system_config
        )
        self.job_monitor = SLURMJobMonitor(self.work_directory)
        self.result_collector = SLURMResultCollector(
            self.work_directory, system_config=self.system_config
        )
        
        logger.info(f"Initialized SLURM manager for {self.system_name} at {self.work_directory}")
        _excl = (self.system_config.get("sbatch_exclude") or "").strip()
        if _excl:
            logger.info("SLURM #SBATCH --exclude=%s (from hpc_config sbatch_exclude)", _excl)
    
    def _setup_working_directory(self) -> None:
        """Set up the working directory structure - only essential directories"""
        directories = [
            "results", 
            "scripts",
        ]
        
        for directory in directories:
            (self.work_directory / directory).mkdir(parents=True, exist_ok=True)

    def _scratch_submit_template(self) -> str:
        """Scratch batch cwd template from hpc_config (may contain ``%u``)."""
        schdir = (self.system_config.get("submit_chdir") or "").strip()
        if schdir:
            return schdir
        art = (self.system_config.get("artifact_dir") or "").strip()
        if art:
            return f"{art.rstrip('/')}/slurm-submit"
        return ""

    def _scratch_submit_host_path(self) -> Optional[Path]:
        tpl = self._scratch_submit_template()
        if not tpl:
            return None
        return slurm_template_to_host_path(tpl.rstrip("/"))

    def _ensure_scratch_slurm_dirs(self) -> None:
        """Create scratch submit/artifact parents so ``#SBATCH --chdir`` succeeds at sbatch time."""
        sch = self._scratch_submit_host_path()
        if sch is not None:
            try:
                (sch / "job_scripts").mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("Could not mkdir scratch job_scripts under %s: %s", sch, e)
        for key in ("submit_chdir", "artifact_dir", "slurm_log_dir"):
            tpl = (self.system_config.get(key) or "").strip()
            if not tpl:
                continue
            try:
                slurm_template_to_host_path(tpl.rstrip("/")).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("Could not mkdir %s path from %s=%r: %s", key, key, tpl, e)

    def stage_job_script_for_submit(self, script_path: Path) -> Path:
        """
        Copy the generated .slurm script to scratch so compute nodes never exec/cat GPFS paths.

        Returns the path to pass to ``sbatch`` (scratch copy when configured, else unchanged).
        """
        src = Path(script_path)
        sch = self._scratch_submit_host_path()
        if sch is None:
            return src
        dest_dir = sch / "job_scripts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        logger.info("Staged job script on scratch for sbatch: %s -> %s", src, dest)
        return dest
    
    def _validate_qos(self) -> None:
        """Validate QOS availability and disable if not available"""
        import subprocess
        
        qos = self.system_config.get("qos")
        if not qos or not qos.strip():
            logger.debug("QOS not specified in system config, skipping QOS validation")
            self.system_config["qos"] = None
            return
        
        # Try to check if QOS is available using sinfo
        try:
            result = subprocess.run(
                ["sinfo", "-o", "%Q", "--noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                available_qos = result.stdout.strip().split()
                if qos not in available_qos:
                    logger.warning(f"QOS '{qos}' not available on this system. Available QOS: {available_qos}. Disabling QOS.")
                    self.system_config["qos"] = None
                else:
                    logger.debug(f"QOS '{qos}' is available on this system")
            else:
                # If sinfo fails, try to disable QOS to avoid submission errors
                logger.warning(f"Could not validate QOS '{qos}'. Disabling QOS to avoid submission errors.")
                self.system_config["qos"] = None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # If sinfo is not available or times out, disable QOS
            logger.warning(f"Could not validate QOS '{qos}' (error: {e}). Disabling QOS to avoid submission errors.")
            self.system_config["qos"] = None
    
    def create_job_config(self, 
                         mpi_ranks: int,
                         omp_threads: int,
                         job_name: str,
                         time_limit: str = "02:00:00",
                         nodes: int = 1,
                         account: str = None,
                         email: str = None,
                         gpus_per_node: int = 0) -> SLURMJobConfig:
        """
        Create a SLURM job configuration for MPI+OpenMP auto-tuning
        
        Args:
            mpi_ranks: Number of MPI ranks per node
            omp_threads: Number of OpenMP threads per rank
            job_name: Name for the job
            time_limit: Time limit in HH:MM:SS format
            nodes: Number of nodes
            account: HPC allocation account
            email: Email for job notifications
            gpus_per_node: Number of GPUs to request per node (default: 0)
            
        Returns:
            SLURMJobConfig object
        """
        if account is None:
            account = self._get_default_account()
        
        # Get QOS from system config, but allow it to be None or empty
        qos = self.system_config.get("qos")
        if qos and not qos.strip():
            qos = None
        
        additional_opts = {}
        launcher = self.system_config.get("launcher")
        if launcher:
            additional_opts["launcher"] = launcher
        # MVAPICH2 + mpiP: Run 1a uses `srun` + LD_PRELOAD (see job_generator); opt out per system with false
        if "mpip_mvapich_srun" in self.system_config:
            additional_opts["mpip_mvapich_srun"] = bool(self.system_config["mpip_mvapich_srun"])
        if self.system_config.get("srun_mpi_flag"):
            additional_opts["srun_mpi_flag"] = str(self.system_config["srun_mpi_flag"]).strip()

        if gpus_per_node > 0:
            # Request GPUs via --gres=gpu:N when the system config opts in.
            # Some systems (TACC Vista/LS6) imply the GPU from the partition itself and throw
            # "Invalid generic resource (gres) specification" if --gres is also specified.
            # Set "gres_gpu": true in hpc_config.json for systems that need the explicit request
            # (e.g. LEAP2 gpu1 partition, most generic GPU clusters).
            if self.system_config.get("gres_gpu", False):
                # Default: match the tuner's GPU count (usually 1). Requesting gpu:8 on nodes that
                # only expose one GPU makes sbatch fail with "Requested node configuration is not available".
                # Some LEAP2 setups reserve full-node CPUs only when all GPUs on an 8-GPU node are
                # requested — set "gpu_gres_count": 8 in hpc_config.json for that site only.
                _goverride = self.system_config.get("gpu_gres_count")
                if _goverride is not None:
                    ngpu = max(1, int(_goverride))
                else:
                    ngpu = max(1, int(gpus_per_node))
                additional_opts["gres"] = f"gpu:{ngpu}"
        
        # Always request exclusive nodes to bypass CPU/GPU ratio limits
        additional_opts["exclusive"] = ""

        # Optional: bad GPFS client / broken nodes (ESTALE on execve). Set in hpc_config.json, e.g. "sbatch_exclude": "gpu1-004"
        _excl = (self.system_config.get("sbatch_exclude") or self.system_config.get("exclude") or "").strip()
        if _excl:
            additional_opts["exclude"] = _excl

        # Batch cwd on scratch — avoids slurmstepd chdir to mmfs1/AutoTuner (ESTALE on some gpu1 nodes).
        _schdir = (self.system_config.get("submit_chdir") or "").strip()
        if not _schdir:
            _art = (self.system_config.get("artifact_dir") or "").strip()
            if _art:
                _schdir = f"{_art.rstrip('/')}/slurm-submit"
        if _schdir:
            additional_opts["chdir"] = _schdir

        # Determine partition based on GPU requirement
        partition_name = self.system_config["partition"]
        if gpus_per_node > 0:
            # Check for GPU partition in config, otherwise use 'gpu' as default
            gpu_partition = self.system_config.get("gpu_partition")
            if gpu_partition:
                partition_name = gpu_partition
            # If no GPU partition in config, try common patterns
            # Disabled fallback to 'gpu' to respect user's explicit partition request (like gh-dev)
            # elif "gpu" not in partition_name.lower():
            #     partition_name = "gpu"  # Generic fallback
            
        env_vars = {
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_SCHEDULE": "dynamic",
            "OMP_DYNAMIC": "TRUE",
        }
        env_vars["OMP_NUM_THREADS"] = str(omp_threads)

        config = SLURMJobConfig(
            job_name=job_name,
            account=account,
            time_limit=time_limit,
            nodes=nodes,
            ntasks_per_node=mpi_ranks,
            cpus_per_task=omp_threads,
            partition=partition_name,
            qos=qos,
            email=email,
            modules=self.system_config["modules"].copy(),
            environment_variables=env_vars,
            additional_options=additional_opts,
        )
        
        # Add MPI-specific environment variables
        if mpi_ranks > 1:
            config.environment_variables.update({
                "MPI_BIND_TO": "core",
                "MPI_MAP_BY": f"socket:PE={omp_threads}",
            })

        # SLURM stdout/stderr on scratch when submit_chdir or slurm_log_dir is set (LEAP2 GPFS ESTALE).
        _schdir_tpl = self._scratch_submit_template()
        _sld = (self.system_config.get("slurm_log_dir") or "").strip()
        if _schdir_tpl:
            logd = _schdir_tpl.rstrip("/")
            config.output_file = f"{logd}/slurm-%j.out"
            config.error_file = f"{logd}/slurm-%j.err"
        elif _sld:
            logd = _sld.rstrip("/")
            config.output_file = f"{logd}/slurm-%j.out"
            config.error_file = f"{logd}/slurm-%j.err"
        else:
            try:
                wd_abs = self.work_directory.expanduser().resolve()
            except OSError:
                wd_abs = self.work_directory.expanduser()
            config.output_file = str(wd_abs / "slurm-%j.out")
            config.error_file = str(wd_abs / "slurm-%j.err")
        
        logger.info(f"Created job config: {job_name} ({mpi_ranks}x{omp_threads})")
        return config
    
    def _get_default_account(self) -> str:
        """Get default HPC account from environment"""
        # Try common environment variable names
        for var in ["HPC_ACCOUNT", "SLURM_ACCOUNT", "SBATCH_ACCOUNT"]:
            account = os.environ.get(var)
            if account:
                return account
        
        # Try to get from SLURM
        try:
            import subprocess
            result = subprocess.run(["sacctmgr", "show", "user", "$USER", "format=account"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
        except:
            pass
        
        # No account found - raise error (user must provide --account or export HPC_ACCOUNT)
        raise ValueError("Could not detect HPC account. Please provide --account argument or export HPC_ACCOUNT/SLURM_ACCOUNT.")
    
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
            profiling: Whether to enable profiling
            likwid_enabled: Whether to enable LIKWID profiling
            likwid_path: Optional path to LIKWID install (e.g. $HOME/software/likwid)
            enable_mpip: If True, try to load mpiP for α (communication).
            mpip_path: Optional path to mpiP install (e.g. $HOME/software/mpiP). Checked first when set.
            light_arguments: Reduced args for multi-rank (Run 1; Run 2 if multi_rank_profile)
            multi_rank_profile: If True, multi-rank jobs also run Run 2 (Nsight wraps full mpirun, all ranks) and export.
            nsight_duration_sec: If set, limit Nsight trace to this many seconds (faster profiling).
            nsight_trace_domains: nsys --trace=... comma list; omit for default osrt,mpi,cuda,cublas. E.g. cuda,cublas for lower overhead.
            spread_nsight_ranks: Deep-finalist flag; Run 2 uses one Nsight session wrapping the full MPI launch (same as phase profiling).
            likwid_rank0_only: If True, LIKWID Run 1b/1c only on MPI rank 0.
            
        Returns:
            Path to the generated job script
        """
        return self.job_generator.generate_job_script(
            job_config, executable_path, arguments, profiling, likwid_enabled, likwid_path, enable_mpip, mpip_path, light_arguments, multi_rank_profile, nsight_duration_sec, nsight_trace_domains, spread_nsight_ranks, likwid_rank0_only,
        )
    
    def submit_job(self, script_path: Path) -> str:
        """
        Submit a SLURM job
        
        Args:
            script_path: Path to the job script
            
        Returns:
            Job ID of the submitted job
        """
        staged = self.stage_job_script_for_submit(Path(script_path))
        submit_cwd = self._scratch_submit_host_path()
        return self.job_monitor.submit_job(staged, submit_cwd=submit_cwd)
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the status of a SLURM job
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            JobStatus object or None if not found
        """
        return self.job_monitor.get_job_status(job_id)
    
    def monitor_jobs(self, job_ids: List[str], check_interval: int = 30) -> Dict[str, str]:
        """
        Monitor multiple jobs until completion
        
        Args:
            job_ids: List of job IDs to monitor
            check_interval: Check interval in seconds
            
        Returns:
            Dictionary mapping job IDs to final states
        """
        return self.job_monitor.monitor_jobs(job_ids, check_interval)
    
    def collect_results(self, job_id: str) -> Path:
        """
        Collect results from a completed job
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Path to results directory
        """
        return self.result_collector.collect_results(job_id, self.job_monitor.job_history)
    
    def submit_batch_jobs(self, 
                          job_configs: List[SLURMJobConfig],
                          executable_path: str,
                          arguments: str = "",
                          profiling: bool = True) -> List[str]:
        """
        Submit multiple jobs in batch
        
        Args:
            job_configs: List of job configurations
            executable_path: Path to the executable
            arguments: Command line arguments
            profiling: Whether to enable profiling
            
        Returns:
            List of submitted job IDs
        """
        logger.info(f"Submitting batch of {len(job_configs)} jobs...")
        
        job_ids = []
        import time
        for i, job_config in enumerate(job_configs):
            retry_count = 0
            max_retries = 30 # roughly 30 * 60s = 30 minutes wait max
            
            # Generate job script
            script_path = None
            try:
                script_path = self.generate_job_script(
                    job_config, executable_path, arguments, profiling
                )
            except Exception as e:
                logger.error(f"Failed to generate job script {i+1}: {e}")
                continue
                
            while retry_count < max_retries:
                try:
                    # Submit job
                    job_id = self.submit_job(script_path)
                    job_ids.append(job_id)
                    
                    logger.info(f"Submitted job {i+1}/{len(job_configs)}: {job_id}")
                    
                    # Small delay between submissions
                    time.sleep(1)
                    break # Success, break retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "QOSMaxSubmitJobPerUser" in error_str or "Job violates accounting/QOS policy" in error_str:
                        logger.warning(f"Hit SLURM submission limit. Waiting 60s before retrying job {i+1}...")
                        
                        # Briefly check on active jobs to encourage updates
                        if job_ids:
                            self.job_monitor.monitor_jobs(job_ids, check_interval=1)
                            
                        time.sleep(60)
                        retry_count += 1
                    else:
                        logger.error(f"Failed to submit job {i+1}: {e}")
                        break # Other error, break retry loop
        
        logger.info(f"Successfully submitted {len(job_ids)} out of {len(job_configs)} jobs")
        return job_ids
    
    def cleanup_jobs(self, job_ids: List[str]) -> None:
        """
        Clean up completed jobs
        
        Args:
            job_ids: List of job IDs to clean up
        """
        self.job_monitor.cleanup_jobs(job_ids)
    
    def get_job_summary(self) -> Dict[str, Any]:
        """
        Get summary of all jobs
        
        Returns:
            Dictionary with job summary information
        """
        summary = self.job_monitor.get_job_summary()
        summary['system_info'] = {
            'system_name': self.system_name,
            'work_directory': str(self.work_directory),
            'system_config': self.system_config
        }
        return summary
    
    def export_job_history(self, output_path: Path) -> None:
        """
        Export job history to files
        
        Args:
            output_path: Directory to save exported data
        """
        self.job_monitor.export_job_history(output_path)
    
    def cleanup_old_results(self, max_age_days: int = 30) -> None:
        """
        Clean up old result directories
        
        Args:
            max_age_days: Maximum age of results to keep (in days)
        """
        self.result_collector.cleanup_old_results(max_age_days)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected results
        
        Returns:
            Dictionary with results summary
        """
        return self.result_collector.get_results_summary()



