#!/usr/bin/env python3
"""
HPC System Configuration Management

Handles system-specific configurations for different HPC systems.
Users can define custom systems via config files or environment variables.
"""

from typing import Dict, Any, Optional
from .models import SystemConfig
import os
import json
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# autotuner/automation/slurm/config.py -> repository root AutoTuner/
_REPO_ROOT = Path(__file__).resolve().parents[3]


class HPCSystemConfig:
    """Manages HPC system-specific configurations (portable, no hardcoded system names)"""
    
    # Default configuration for unknown systems
    DEFAULT_CONFIG = SystemConfig(
        partition="normal",
        qos=None,
        max_nodes=100,
        max_time="48:00:00",
        modules=[],  # User must load appropriate modules
        features=[]
    )
    
    @staticmethod
    def get_system_configs() -> Dict[str, SystemConfig]:
        """
        Get all available HPC system configurations.
        
        Priority:
        1. User-defined config file (config/hpc_config.json, repo root, or ~/.config/autotuner/)
        2. Environment variables (HPC_PARTITION, HPC_MODULES, etc.)
        3. Built-in defaults (minimal, portable)
        """
        # Try to load user config file
        user_configs = HPCSystemConfig._load_user_config()
        if user_configs:
            return user_configs
        
        # Return minimal default config that works on most SLURM systems
        return {
            "default": HPCSystemConfig.DEFAULT_CONFIG
        }
    
    @staticmethod
    def _load_user_config() -> Optional[Dict[str, SystemConfig]]:
        """Load user-defined HPC configuration from JSON file"""
        config_locations = [
            Path.cwd() / "config" / "hpc_config.json",
            _REPO_ROOT / "config" / "hpc_config.json",
            Path.cwd() / "hpc_config.json",
            _REPO_ROOT / "hpc_config.json",
            Path.home() / ".config" / "autotuner" / "hpc_config.json",
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                    
                    configs = {}
                    # Support both flat layout and wrapped {"systems": {...}} layout
                    systems_dict = data.get("systems", data)
                    
                    for name, cfg in systems_dict.items():
                        if name.startswith("_"):
                            continue
                        if not isinstance(cfg, dict):
                            continue
                        configs[name] = SystemConfig(
                            partition=cfg.get("partition", "normal"),
                            qos=cfg.get("qos"),
                            max_nodes=cfg.get("max_nodes", 100),
                            max_time=cfg.get("max_time", "48:00:00"),
                            modules=cfg.get("modules", []),
                            features=cfg.get("features", []),
                            gpu_type=cfg.get("gpu_type"),
                            gpus_per_node=cfg.get("gpus_per_node"),
                            launcher=cfg.get("launcher"),
                            mpip_mvapich_srun=cfg.get("mpip_mvapich_srun"),
                            srun_mpi_flag=cfg.get("srun_mpi_flag"),
                            gpu_partition=cfg.get("gpu_partition"),
                            gres_gpu=cfg.get("gres_gpu"),
                            gpu_gres_count=(
                                int(cfg["gpu_gres_count"])
                                if cfg.get("gpu_gres_count") is not None
                                else None
                            ),
                            sbatch_exclude=cfg.get("sbatch_exclude"),
                            slurm_log_dir=cfg.get("slurm_log_dir"),
                            artifact_dir=cfg.get("artifact_dir"),
                            submit_chdir=cfg.get("submit_chdir"),
                        )
                    return configs if configs else None
                except Exception:
                    pass
        
        return None
    
    @staticmethod
    def _detect_slurm_partition() -> Optional[str]:
        """Detect default SLURM partition using sinfo"""
        try:
            # Get partition names (default has *)
            result = subprocess.check_output(["sinfo", "-h", "-o", "%P"], text=True)
            partitions = [p.strip() for p in result.splitlines() if p.strip()]
            
            # Look for default partition (marked with *)
            for p in partitions:
                if p.endswith("*"):
                    return p.rstrip("*")
            
            # If no default marked, return the first available one
            if partitions:
                return partitions[0].rstrip("*")
                
        except (subprocess.SubprocessError, FileNotFoundError):
            # sinfo not found or failed
            pass
        return None

    @staticmethod
    def get_system_config(system_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific system as dictionary.
        
        If system_name is not found, uses environment variables or defaults.
        """
        configs = HPCSystemConfig.get_system_configs()
        
        # Try exact match
        if system_name and system_name.lower() in configs:
            system_config = configs[system_name.lower()]
            
            # Allow environment overrides (important for CLI --partition arg)
            if os.environ.get("HPC_PARTITION"):
                import dataclasses
                system_config = dataclasses.replace(system_config, partition=os.environ.get("HPC_PARTITION"))
                
        # Try to find from environment (only if not found in configs above)
        elif os.environ.get("HPC_PARTITION"):
            sn = (system_name or "").strip().lower()
            if sn and sn not in configs and sn != "default":
                logger.warning(
                    "HPC system %r is not defined in hpc_config.json (loaded keys: %s). "
                    "Using partition from HPC_PARTITION only — job scripts will have **no module loads** "
                    "unless HPC_MODULES is set. Copy config/hpc_config.example.json to config/hpc_config.json "
                    "and add a %r block with necessary modules.",
                    system_name,
                    sorted(configs.keys()),
                    sn,
                )
            # --partition sets HPC_PARTITION; without --system <key> matching hpc_config.json, modules stay empty.
            hm_preview = [m.strip() for m in os.environ.get("HPC_MODULES", "").split(",") if m.strip()]
            if not hm_preview and sn in ("", "default"):
                logger.warning(
                    "HPC_PARTITION is set but system name is %r and HPC_MODULES is empty — job scripts will load "
                    "**no** modules. Use e.g. --system ls6 with "
                    "hpc_config.json, or export HPC_MODULES=gcc/11.2.0,cuda/12.8,mvapich2-gdr/2.3.7,python3/3.9.7 "
                    "(comma-separated). Config keys loaded: %s",
                    sn or "default",
                    sorted(configs.keys()),
                )
            system_config = SystemConfig(
                partition=os.environ.get("HPC_PARTITION", "normal"),
                qos=os.environ.get("HPC_QOS"),
                max_nodes=int(os.environ.get("HPC_MAX_NODES", "100")),
                max_time=os.environ.get("HPC_MAX_TIME", "48:00:00"),
                modules=[m.strip() for m in os.environ.get("HPC_MODULES", "").split(",") if m.strip()],
                features=[f.strip() for f in os.environ.get("HPC_FEATURES", "").split(",") if f.strip()],
                gpu_type=os.environ.get("HPC_GPU_TYPE"),
                gpus_per_node=int(os.environ.get("HPC_GPUS_PER_NODE", "0")) if os.environ.get("HPC_GPUS_PER_NODE") else None
            )
        # Use default with auto-detection
        else:
            system_config = configs.get("default", HPCSystemConfig.DEFAULT_CONFIG)
            
            # If using default 'normal' which might be wrong on this system, try to detect
            if system_config.partition == "normal":
                detected = HPCSystemConfig._detect_slurm_partition()
                if detected and detected != "normal":
                    # Create a copy with the detected partition
                    import dataclasses
                    system_config = dataclasses.replace(system_config, partition=detected)
        
        # Convert to dictionary format for backward compatibility
        config_dict = {
            "partition": system_config.partition,
            "qos": system_config.qos,
            "max_nodes": system_config.max_nodes,
            "max_time": system_config.max_time,
            "modules": system_config.modules,
            "features": system_config.features
        }
        
        if system_config.gpu_type:
            config_dict["gpu_type"] = system_config.gpu_type
        if system_config.gpus_per_node:
            config_dict["gpus_per_node"] = system_config.gpus_per_node
        if system_config.launcher:
            config_dict["launcher"] = system_config.launcher
        if getattr(system_config, "mpip_mvapich_srun", None) is not None:
            config_dict["mpip_mvapich_srun"] = bool(system_config.mpip_mvapich_srun)
        sf = getattr(system_config, "srun_mpi_flag", None)
        if sf:
            config_dict["srun_mpi_flag"] = str(sf).strip()
        gp = getattr(system_config, "gpu_partition", None)
        if gp:
            config_dict["gpu_partition"] = str(gp).strip()
        if getattr(system_config, "gres_gpu", None) is not None:
            config_dict["gres_gpu"] = bool(system_config.gres_gpu)
        ggc = getattr(system_config, "gpu_gres_count", None)
        if ggc is not None:
            config_dict["gpu_gres_count"] = int(ggc)
        bx = getattr(system_config, "sbatch_exclude", None)
        if bx:
            config_dict["sbatch_exclude"] = str(bx).strip()
        sld = getattr(system_config, "slurm_log_dir", None)
        if sld:
            config_dict["slurm_log_dir"] = str(sld).strip()
        ad = getattr(system_config, "artifact_dir", None)
        if ad:
            config_dict["artifact_dir"] = str(ad).strip()
        scd = getattr(system_config, "submit_chdir", None)
        if scd:
            config_dict["submit_chdir"] = str(scd).strip()

        # Optional env overrides (after JSON/default resolution): same keys as hpc_config.json
        _launcher_env = os.environ.get("HPC_LAUNCHER", "").strip()
        if _launcher_env:
            config_dict["launcher"] = _launcher_env
        _mv_env = os.environ.get("HPC_MPIP_MVAPICH_SRUN", "").strip().lower()
        if _mv_env in ("1", "true", "yes", "on"):
            config_dict["mpip_mvapich_srun"] = True
        elif _mv_env in ("0", "false", "no", "off"):
            config_dict["mpip_mvapich_srun"] = False
        _srun_flag_env = os.environ.get("HPC_SRUN_MPI_FLAG", "").strip()
        if _srun_flag_env:
            config_dict["srun_mpi_flag"] = _srun_flag_env

        return config_dict