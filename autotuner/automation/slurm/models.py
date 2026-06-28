#!/usr/bin/env python3
"""
Data models for SLURM job management

Contains data classes and structures used throughout the SLURM management system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SLURMJobConfig:
    """SLURM job configuration parameters"""
    job_name: str
    account: str
    time_limit: str
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    mem_per_node: str = "16G"
    partition: str = "normal"
    qos: Optional[str] = "normal"
    email: Optional[str] = None
    mail_type: str = "END,FAIL"
    output_file: str = "slurm-%j.out"
    error_file: str = "slurm-%j.err"
    working_directory: str = "."
    modules: List[str] = None
    environment_variables: Dict[str, str] = None
    additional_options: Dict[str, str] = None
    # OpenMPI mpirun --map-by ... pe=<N> (not emitted as #SBATCH; optional per-job override)
    launcher_pe_per_rank: Optional[str] = None

    def __post_init__(self):
        """Set default values after initialization"""
        if self.modules is None:
            self.modules = ["gcc", "openmpi", "cuda"]
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.additional_options is None:
            self.additional_options = {}


@dataclass
class JobStatus:
    """Job status information"""
    job_id: str
    state: str
    submit_time: str
    start_time: str
    end_time: str
    runtime: str
    exit_code: str
    nodes: str
    cpus: str
    memory: str


@dataclass
class SystemConfig:
    """HPC system configuration"""
    partition: str
    qos: Optional[str]
    max_nodes: int
    max_time: str
    modules: List[str]
    features: List[str]
    gpu_type: Optional[str] = None
    gpus_per_node: Optional[int] = None
    launcher: Optional[str] = None
    mpip_mvapich_srun: Optional[bool] = None
    srun_mpi_flag: Optional[str] = None
    gpu_partition: Optional[str] = None
    gres_gpu: Optional[bool] = None
    gpu_gres_count: Optional[int] = None
    sbatch_exclude: Optional[str] = None
    slurm_log_dir: Optional[str] = None
    artifact_dir: Optional[str] = None
    submit_chdir: Optional[str] = None


@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    status: str
    results_directory: Optional[str] = None
    profiling_data: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

