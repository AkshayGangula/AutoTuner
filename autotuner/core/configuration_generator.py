#!/usr/bin/env python3
"""
Complete Configuration Generator for MPI+OpenMP Auto-Tuning

Generates candidate MPI+OpenMP configurations based on complete hardware topology analysis
from TARGET_INFO_SYSTEM_ENV, including NUMA-aware configurations, power-of-2 strategies, 
and scaling approaches using ALL tuples from ALL tables.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HardwareTopology:
    """Hardware topology information"""
    total_cores: int
    sockets: int
    cores_per_socket: int
    numa_domains: int
    cores_per_numa: int
    memory_per_node_gb: float
    architecture: str
    system_name: str
    
    def __str__(self):
        return (f"{self.system_name}: {self.total_cores} cores, "
                f"{self.sockets} sockets, {self.numa_domains} NUMA domains")
    
    @classmethod
    def from_complete_dataset_analysis(cls, hardware_config: Dict[str, Any]) -> 'HardwareTopology':
        """
        Create HardwareTopology from complete dataset analysis using ALL TARGET_INFO_SYSTEM_ENV tuples
        
        Args:
            hardware_config: Complete hardware configuration from dataset analysis
            
        Returns:
            HardwareTopology object with complete hardware information
        """
        cpu_cores = hardware_config.get('cpu_cores', 0)
        cpu_speed_mhz = hardware_config.get('cpu_speed_mhz', 0)
        cpu_architecture = hardware_config.get('cpu_architecture', 'unknown')
        cpu_description = hardware_config.get('cpu_description', 'Unknown CPU')
        
        # Extract detailed CPU information from JSON if available
        cpu_info_json = hardware_config.get('cpu_info_json', {})
        sockets = 1  # Default
        cores_per_socket = cpu_cores  # Default
        
        if cpu_info_json and 'items' in cpu_info_json:
            # Analyze CPU info JSON to determine topology
            cpu_items = cpu_info_json['items']
            if cpu_items:
                # Count unique sockets
                sockets = len(set(item.get('socket', 0) for item in cpu_items))
                cores_per_socket = cpu_cores // max(sockets, 1)
        
        # Estimate NUMA domains based on CPU cores and architecture
        numa_domains = cls._estimate_numa_domains(cpu_cores, cpu_architecture, sockets)
        cores_per_numa = cpu_cores // max(numa_domains, 1)
        
        # Estimate memory based on architecture and core count
        memory_per_node_gb = cls._estimate_memory_per_node(cpu_cores, cpu_architecture)
        
        return cls(
            total_cores=cpu_cores,
            sockets=sockets,
            cores_per_socket=cores_per_socket,
            numa_domains=numa_domains,
            cores_per_numa=cores_per_numa,
            memory_per_node_gb=memory_per_node_gb,
            architecture=cpu_architecture,
            system_name=f"{cpu_description} ({cpu_architecture})"
        )
    
    @staticmethod
    def _estimate_numa_domains(cpu_cores: int, architecture: str, sockets: int) -> int:
        """Estimate NUMA domains based on hardware characteristics"""
        # AMD EPYC processors typically have 2 NUMA domains per socket
        if 'EPYC' in architecture.upper() or 'AMD' in architecture.upper():
            return max(2, sockets * 2)
        
        # Intel Xeon processors typically have 1-2 NUMA domains per socket
        elif 'XEON' in architecture.upper() or 'INTEL' in architecture.upper():
            return max(1, sockets)
        
        # Default estimation based on core count
        elif cpu_cores >= 64:
            return max(4, cpu_cores // 16)  # High-core systems
        elif cpu_cores >= 32:
            return max(2, cpu_cores // 16)  # Medium-core systems
        else:
            return 1  # Low-core systems
    
    @staticmethod
    def _estimate_memory_per_node(cpu_cores: int, architecture: str) -> float:
        """Estimate memory per node based on hardware characteristics"""
        # AMD EPYC systems typically have high memory capacity
        if 'EPYC' in architecture.upper() or 'AMD' in architecture.upper():
            if cpu_cores >= 64:
                return 512.0  # High-end EPYC systems
            elif cpu_cores >= 32:
                return 256.0  # Mid-range EPYC systems
            else:
                return 128.0  # Lower-end EPYC systems
        
        # Intel Xeon systems
        elif 'XEON' in architecture.upper() or 'INTEL' in architecture.upper():
            if cpu_cores >= 64:
                return 256.0  # High-end Xeon systems
            elif cpu_cores >= 32:
                return 128.0  # Mid-range Xeon systems
            else:
                return 64.0   # Lower-end Xeon systems
        
        # Default estimation
        else:
            return max(64.0, cpu_cores * 2.0)  # 2GB per core estimate

@dataclass
class MPIOpenMPConfig:
    """MPI+OpenMP configuration specification"""
    name: str
    mpi_ranks_per_node: int
    omp_threads_per_rank: int
    binding_strategy: str
    placement_strategy: str
    numa_policy: str
    description: str
    # If set, SLURM --nodes for this job (LULESH may use 1..max_nodes sweep). None = tuner's global --nodes.
    num_nodes: Optional[int] = None

    def __str__(self):
        return f"{self.name} ({self.mpi_ranks_per_node}x{self.omp_threads_per_rank})"
    
    def get_slurm_config(self) -> Dict[str, str]:
        """Get SLURM configuration parameters"""
        return {
            'ntasks_per_node': str(self.mpi_ranks_per_node),
            'cpus_per_task': str(self.omp_threads_per_rank),
            'bind_to': self.binding_strategy,
            'distribution': self.placement_strategy
        }
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get OpenMP environment variables"""
        return {
            'OMP_NUM_THREADS': str(self.omp_threads_per_rank),
            'OMP_PROC_BIND': self.binding_strategy,
            'OMP_PLACES': self.placement_strategy,
            'OMP_SCHEDULE': 'dynamic',
            'OMP_DYNAMIC': 'TRUE'
        }

class ConfigurationGenerator:
    """
    Generates MPI+OpenMP configurations based on hardware topology
    
    Implements multiple strategies:
    1. Power-of-2 configurations
    2. NUMA-aware configurations
    3. Socket-aware configurations
    4. Scaling configurations (weak/strong)
    """
    
    def __init__(self, hardware: HardwareTopology):
        """
        Initialize with hardware topology
        
        Args:
            hardware: Hardware topology information
        """
        self.hardware = hardware
        self.generated_configs = []
        
        logger.info(f"Initialized configuration generator for {hardware}")
    
    @classmethod
    def from_complete_dataset_analysis(cls, hardware_config: Dict[str, Any]) -> 'ConfigurationGenerator':
        """
        Create ConfigurationGenerator from complete dataset analysis using ALL TARGET_INFO_SYSTEM_ENV tuples
        
        Args:
            hardware_config: Complete hardware configuration from dataset analysis
            
        Returns:
            ConfigurationGenerator with complete hardware topology
        """
        # Create hardware topology from complete dataset analysis
        hardware_topology = HardwareTopology.from_complete_dataset_analysis(hardware_config)
        
        # Create configuration generator
        generator = cls(hardware_topology)
        
        logger.info(f"Created ConfigurationGenerator from complete dataset analysis: {hardware_topology}")
        return generator
    
    def generate_hardware_aware_configurations(self, hardware_config: Dict[str, Any]) -> List[MPIOpenMPConfig]:
        """
        Generate configurations using complete hardware analysis from dataset
        
        Args:
            hardware_config: Complete hardware configuration from dataset analysis
            
        Returns:
            List of hardware-aware MPI+OpenMP configurations
        """
        logger.info("Generating hardware-aware configurations from complete dataset analysis...")
        
        configs = []
        cpu_cores = hardware_config.get('cpu_cores', 0)
        cpu_architecture = hardware_config.get('cpu_architecture', 'unknown')
        cpu_description = hardware_config.get('cpu_description', 'Unknown CPU')
        
        # Get available configurations from hardware constraints
        hardware_constraints = hardware_config.get('hardware_constraints', {})
        available_configs = hardware_constraints.get('available_configurations', [])
        
        # Use pre-computed configurations from dataset analysis
        for config_info in available_configs:
            mpi_ranks = config_info.get('mpi_ranks', 1)
            openmp_threads = config_info.get('openmp_threads', cpu_cores)
            description = config_info.get('description', 'Hardware-aware configuration')
            suitability = config_info.get('suitability', 'General purpose')
            
            # Determine binding strategy based on configuration
            if mpi_ranks <= self.hardware.sockets:
                binding = "socket"
                placement = "sockets"
            elif mpi_ranks <= self.hardware.numa_domains:
                binding = "numa"
                placement = "numa"
            else:
                binding = "core"
                placement = "cores"
            
            config = self._create_config(
                mpi_ranks, openmp_threads, binding, placement, "first-touch"
            )
            config.description = f"{description} - {suitability}"
            configs.append(config)
        
        # Add additional configurations based on complete hardware analysis
        configs.extend(self._generate_architecture_specific_configs(cpu_architecture, cpu_cores))
        
        logger.info(f"Generated {len(configs)} hardware-aware configurations from complete dataset analysis")
        return configs
    
    def _generate_architecture_specific_configs(self, architecture: str, cpu_cores: int) -> List[MPIOpenMPConfig]:
        """
        Generate architecture-specific configurations based on complete hardware analysis
        
        Args:
            architecture: CPU architecture from dataset analysis
            cpu_cores: Number of CPU cores from dataset analysis
            
        Returns:
            List of architecture-specific configurations
        """
        configs = []
        
        # AMD EPYC specific configurations
        if 'EPYC' in architecture.upper() or 'AMD' in architecture.upper():
            # EPYC processors benefit from NUMA-aware configurations
            numa_configs = [
                (2, cpu_cores // 2),      # 2 NUMA domains
                (4, cpu_cores // 4),      # 4 NUMA domains
                (8, cpu_cores // 8),      # 8 NUMA domains
            ]
            
            for mpi_ranks, omp_threads in numa_configs:
                if omp_threads > 0 and mpi_ranks <= cpu_cores:
                    config = self._create_config(
                        mpi_ranks, omp_threads, "numa", "numa", "first-touch"
                    )
                    config.description = f"AMD EPYC NUMA-optimized ({mpi_ranks}x{omp_threads})"
                    configs.append(config)
        
        # Intel Xeon specific configurations
        elif 'XEON' in architecture.upper() or 'INTEL' in architecture.upper():
            # Xeon processors benefit from socket-aware configurations
            socket_configs = [
                (1, cpu_cores),           # Single socket
                (2, cpu_cores // 2),      # Two sockets
                (4, cpu_cores // 4),      # Four sockets
            ]
            
            for mpi_ranks, omp_threads in socket_configs:
                if omp_threads > 0 and mpi_ranks <= cpu_cores:
                    config = self._create_config(
                        mpi_ranks, omp_threads, "socket", "sockets", "first-touch"
                    )
                    config.description = f"Intel Xeon socket-optimized ({mpi_ranks}x{omp_threads})"
                    configs.append(config)
        
        return configs
    
    def generate_all_configurations(self) -> List[MPIOpenMPConfig]:
        """
        Generate all possible valid configurations
        
        Returns:
            List of all valid MPI+OpenMP configurations
        """
        logger.info("Generating all possible configurations...")
        
        configs = []
        
        # Generate configurations for all valid MPI rank counts
        for mpi_ranks in range(1, self.hardware.total_cores + 1):
            if self.hardware.total_cores % mpi_ranks == 0:
                omp_threads = self.hardware.total_cores // mpi_ranks
                
                config = self._create_config(
                    mpi_ranks, omp_threads, "balanced", "cores", "interleave"
                )
                configs.append(config)
        
        self.generated_configs = configs
        logger.info(f"Generated {len(configs)} total configurations")
        
        return configs
    
    def generate_paper_configurations(self) -> List[MPIOpenMPConfig]:
        """
        Generate configurations following the research paper methodology
        
        Focuses on balanced configurations with power-of-2 relationships
        Dynamically generates power-of-2 configurations based on hardware topology
        """
        logger.info("Generating paper methodology configurations...")
        
        configs = []
        
        # Dynamically generate power-of-2 MPI ranks based on hardware
        # This scales with system size instead of hardcoding limits
        mpi_ranks_options = []
        power = 0
        while True:
            ranks = 2 ** power
            if ranks > self.hardware.total_cores:
                break
            mpi_ranks_options.append(ranks)
            power += 1
        
        # Ensure we have at least the minimal configurations
        if not mpi_ranks_options:
            mpi_ranks_options = [1]
        
        for mpi_ranks in mpi_ranks_options:
            if self.hardware.total_cores % mpi_ranks == 0:
                omp_threads = self.hardware.total_cores // mpi_ranks
                
                # Determine binding strategy based on configuration
                if mpi_ranks <= self.hardware.sockets:
                    binding = "socket"
                    placement = "sockets"
                elif mpi_ranks <= self.hardware.numa_domains:
                    binding = "numa"
                    placement = "numa"
                else:
                    binding = "core"
                    placement = "cores"
                
                config = self._create_config(
                    mpi_ranks, omp_threads, binding, placement, "interleave"
                )
                configs.append(config)
        
        # Add pure MPI configuration (total_cores x 1) if not already included
        # This ensures we test pure MPI even if total_cores is not a power of 2
        # Check if pure MPI already exists (any config with omp_threads == 1 and mpi_ranks == total_cores)
        has_pure_mpi = any(
            c.omp_threads_per_rank == 1 and c.mpi_ranks_per_node == self.hardware.total_cores 
            for c in configs
        )
        
        if not has_pure_mpi:
            pure_mpi_config = self._create_config(
                self.hardware.total_cores, 1, "core", "cores", "interleave",
                "Pure MPI configuration - one thread per MPI rank"
            )
            configs.append(pure_mpi_config)
            logger.info(f"Added pure MPI configuration: {pure_mpi_config.name} ({self.hardware.total_cores} MPI ranks, 1 thread/rank)")
        
        logger.info(f"Generated {len(configs)} paper methodology configurations")
        return configs

    def generate_lulesh_configurations(self, max_nodes: int) -> List[MPIOpenMPConfig]:
        """
        Configurations valid for LLNL LULESH 2.x: total MPI ranks must be a perfect cube
        (1, 8, 27, 64, …). See InitMeshDecomp() in lulesh-init.cc.

        Sweeps SLURM node counts N = 1 .. max_nodes (--nodes). For each N, adds every
        layout where T = N * mpi_ranks_per_node = k³ and C % mpi_ranks_per_node == 0.

        Example: --nodes 2 on 72-core nodes yields three jobs: 1×72 and 8×9 on 1 node,
        and 4×18 on 2 nodes (all cube-valid).
        """
        C = self.hardware.total_cores
        max_n = max(1, int(max_nodes))
        configs: List[MPIOpenMPConfig] = []
        seen_keys: set = set()

        for N in range(1, max_n + 1):
            max_total = N * C
            k = 1
            while True:
                T = k * k * k
                if T > max_total:
                    break
                if T % N != 0:
                    k += 1
                    continue
                R = T // N
                if R < 1 or R > C or (C % R) != 0:
                    k += 1
                    continue
                omp_threads = C // R
                key = (N, R, omp_threads)
                if key in seen_keys:
                    k += 1
                    continue
                seen_keys.add(key)
                if R <= self.hardware.sockets:
                    binding, placement = "socket", "sockets"
                elif R <= self.hardware.numa_domains:
                    binding, placement = "numa", "numa"
                else:
                    binding, placement = "core", "cores"
                cfg = self._create_config(
                    R,
                    omp_threads,
                    binding,
                    placement,
                    "interleave",
                    f"LULESH: k³={T} total ranks on {N} node(s) ({k}³, {R} ranks/node × {omp_threads} threads)",
                    job_num_nodes=N,
                )
                configs.append(cfg)
                k += 1

        configs.sort(key=lambda c: (c.num_nodes or 0, c.mpi_ranks_per_node, c.omp_threads_per_rank))
        logger.info(
            f"LULESH: generated {len(configs)} cube-valid job(s) "
            f"(sweep N=1..{max_n} nodes, {C} cores/node)"
        )
        if len(configs) == 0:
            logger.error(
                "LULESH: no valid configurations — increase --nodes "
                "(e.g. 3 nodes → 27 ranks / 9×8; 4 nodes → 8 and 64 total ranks)."
            )
        return configs

    def generate_minimd_configurations(self) -> List[MPIOpenMPConfig]:
        """
        Mantevo miniMD: same sweep as paper + NUMA-aware, but omit full-node pure MPI
        (``total_cores``×1). That layout often stalls or hits SLURM time limits on
        typical benchmark inputs while other decompositions finish.
        """
        paper = self.generate_paper_configurations()
        numa = self.generate_numa_aware_configurations()
        C = self.hardware.total_cores
        out: List[MPIOpenMPConfig] = []
        seen: set = set()
        skipped_full_pure = False
        for config in paper + numa:
            if config.name in seen:
                continue
            if config.mpi_ranks_per_node == C and config.omp_threads_per_rank == 1:
                skipped_full_pure = True
                continue
            seen.add(config.name)
            out.append(config)
        if skipped_full_pure:
            logger.info(
                f"miniMD: skipped {C}x1 full-node pure MPI (often times out); "
                f"{len(out)} configuration(s) after merge/dedupe"
            )
        else:
            logger.info(f"miniMD: {len(out)} configuration(s) after merge/dedupe")
        return out

    def generate_numa_aware_configurations(self) -> List[MPIOpenMPConfig]:
        """
        Generate NUMA-aware configurations
        
        Optimizes for memory locality and NUMA domain alignment
        """
        logger.info("Generating NUMA-aware configurations...")
        
        configs = []
        
        # NUMA-aware configurations
        numa_configs = [
            # One MPI rank per NUMA domain
            (self.hardware.numa_domains, self.hardware.cores_per_numa),
            # Two MPI ranks per NUMA domain
            (self.hardware.numa_domains * 2, self.hardware.cores_per_numa // 2),
            # Four MPI ranks per NUMA domain
            (self.hardware.numa_domains * 4, self.hardware.cores_per_numa // 4),
        ]
        
        for mpi_ranks, omp_threads in numa_configs:
            if omp_threads > 0 and mpi_ranks <= self.hardware.total_cores:
                config = self._create_config(
                    mpi_ranks, omp_threads, "numa", "numa", "first-touch"
                )
                configs.append(config)
        
        # Socket-aware configurations
        socket_configs = [
            # One MPI rank per socket
            (self.hardware.sockets, self.hardware.cores_per_socket),
            # Two MPI ranks per socket
            (self.hardware.sockets * 2, self.hardware.cores_per_socket // 2),
        ]
        
        for mpi_ranks, omp_threads in socket_configs:
            if omp_threads > 0 and mpi_ranks <= self.hardware.total_cores:
                config = self._create_config(
                    mpi_ranks, omp_threads, "socket", "sockets", "first-touch"
                )
                configs.append(config)
        
        logger.info(f"Generated {len(configs)} NUMA-aware configurations")
        return configs
    
    def generate_scaling_configurations(self, 
                                     base_problem_size: int,
                                     scaling_factors: List[float] = None) -> List[MPIOpenMPConfig]:
        """
        Generate configurations for scaling studies
        
        Args:
            base_problem_size: Base problem size for scaling
            scaling_factors: List of scaling factors (default: [1, 2, 4, 8])
            
        Returns:
            List of scaling configurations
        """
        if scaling_factors is None:
            scaling_factors = [1.0, 2.0, 4.0, 8.0]
        
        logger.info(f"Generating scaling configurations for problem size {base_problem_size}")
        
        configs = []
        
        # Use recommended configurations for scaling
        base_configs = self.get_recommended_configurations()
        
        for config in base_configs:
            for factor in scaling_factors:
                # Create scaling variant
                scaled_config = MPIOpenMPConfig(
                    name=f"{config.name}_scale_{factor}",
                    mpi_ranks_per_node=config.mpi_ranks_per_node,
                    omp_threads_per_rank=config.omp_threads_per_rank,
                    binding_strategy=config.binding_strategy,
                    placement_strategy=config.placement_strategy,
                    numa_policy=config.numa_policy,
                    description=f"{config.description} (scaling factor: {factor})",
                    num_nodes=config.num_nodes,
                )
                configs.append(scaled_config)
        
        logger.info(f"Generated {len(configs)} scaling configurations")
        return configs
    
    def get_recommended_configurations(self) -> List[MPIOpenMPConfig]:
        """
        Get recommended configurations for pilot mode (subset of power-of-2 configs)
        
        Returns:
            List of recommended configurations (subset for pilot mode)
        """
        logger.info("Generating recommended configurations (pilot mode)...")
        
        # For pilot mode, return only a subset of power-of-2 configurations
        # This ensures pilot mode has fewer configs than full mode
        power_configs = self.generate_paper_configurations()
        
        # Return only key power-of-2 configs: first, middle, and last
        # For 48 cores: 1x48, 2x24, 4x12, 8x6, 16x3, 48x1
        # Pilot subset: 1x48, 4x12, 48x1 (3 key configs) or 1x48, 2x24, 8x6, 48x1 (4 configs)
        # Let's use: 1x48, 2x24, 8x6, 48x1 (4 essential configs covering the range)
        configs = []
        
        if len(power_configs) >= 1:
            configs.append(power_configs[0])  # First: 1x48 (pure OpenMP)
        
        if len(power_configs) >= 2:
            configs.append(power_configs[1])  # Second: 2x24
        
        # Middle config (approximately)
        if len(power_configs) >= 4:
            configs.append(power_configs[3])  # Fourth: 8x6
        
        # Last config (pure MPI)
        if len(power_configs) >= 2:
            configs.append(power_configs[-1])  # Last: 48x1 (pure MPI)
        
        logger.info(f"Generated {len(configs)} recommended configurations (pilot mode)")
        return configs
    
    def _create_config(self, 
                      mpi_ranks: int, 
                      omp_threads: int,
                      binding: str,
                      placement: str,
                      numa_policy: str,
                      description: str = "",
                      job_num_nodes: Optional[int] = None) -> MPIOpenMPConfig:
        """
        Create a configuration object
        
        Args:
            mpi_ranks: Number of MPI ranks per node
            omp_threads: Number of OpenMP threads per rank
            binding: Thread binding strategy
            placement: Thread placement strategy
            numa_policy: NUMA memory policy
            description: Configuration description
            
        Returns:
            MPIOpenMPConfig object
        """
        name = f"{mpi_ranks}x{omp_threads}"
        
        if not description:
            if mpi_ranks == 1:
                description = "Pure OpenMP configuration"
            elif omp_threads == 1:
                description = "Pure MPI configuration"
            else:
                description = f"Hybrid MPI+OpenMP: {mpi_ranks} ranks × {omp_threads} threads"
        
        config = MPIOpenMPConfig(
            name=name,
            mpi_ranks_per_node=mpi_ranks,
            omp_threads_per_rank=omp_threads,
            binding_strategy=binding,
            placement_strategy=placement,
            numa_policy=numa_policy,
            description=description,
            num_nodes=job_num_nodes,
        )
        
        return config
    
    def validate_configuration(self, config: MPIOpenMPConfig) -> bool:
        """
        Validate a configuration against hardware constraints
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check total core usage
        total_cores_used = config.mpi_ranks_per_node * config.omp_threads_per_rank
        
        if total_cores_used > self.hardware.total_cores:
            logger.warning(f"Configuration {config.name} uses {total_cores_used} cores, "
                          f"but only {self.hardware.total_cores} available")
            return False
        
        # Check MPI rank constraints
        if config.mpi_ranks_per_node < 1:
            logger.warning(f"Configuration {config.name} has invalid MPI rank count")
            return False
        
        # Check OpenMP thread constraints
        if config.omp_threads_per_rank < 1:
            logger.warning(f"Configuration {config.name} has invalid OpenMP thread count")
            return False
        
        # Check binding strategy compatibility
        if config.binding_strategy == "socket" and config.mpi_ranks_per_node > self.hardware.sockets:
            logger.warning(f"Configuration {config.name} uses socket binding but has more ranks than sockets")
            return False
        
        if config.binding_strategy == "numa" and config.mpi_ranks_per_node > self.hardware.numa_domains:
            logger.warning(f"Configuration {config.name} uses NUMA binding but has more ranks than NUMA domains")
            return False
        
        return True
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of all generated configurations
        
        Returns:
            Dictionary with configuration summary
        """
        if not self.generated_configs:
            return {"message": "No configurations generated yet"}
        
        summary = {
            "total_configurations": len(self.generated_configs),
            "hardware_info": {
                "total_cores": self.hardware.total_cores,
                "sockets": self.hardware.sockets,
                "numa_domains": self.hardware.numa_domains,
                "system_name": self.hardware.system_name
            },
            "configuration_types": {
                "pure_mpi": len([c for c in self.generated_configs if c.omp_threads_per_rank == 1]),
                "pure_openmp": len([c for c in self.generated_configs if c.mpi_ranks_per_node == 1]),
                "hybrid": len([c for c in self.generated_configs if c.mpi_ranks_per_node > 1 and c.omp_threads_per_rank > 1])
            },
            "mpi_rank_distribution": {},
            "omp_thread_distribution": {}
        }
        
        # Analyze MPI rank distribution
        for config in self.generated_configs:
            mpi_ranks = config.mpi_ranks_per_node
            summary["mpi_rank_distribution"][mpi_ranks] = summary["mpi_rank_distribution"].get(mpi_ranks, 0) + 1
        
        # Analyze OpenMP thread distribution
        for config in self.generated_configs:
            omp_threads = config.omp_threads_per_rank
            summary["omp_thread_distribution"][omp_threads] = summary["omp_thread_distribution"].get(omp_threads, 0) + 1
        
        return summary
    
    def export_configurations(self, output_path: Path) -> None:
        """
        Export configurations to various formats
        
        Args:
            output_path: Directory to save exported configurations
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export to JSON
        import json
        configs_data = []
        
        for config in self.generated_configs:
            config_data = {
                'name': config.name,
                'mpi_ranks_per_node': config.mpi_ranks_per_node,
                'omp_threads_per_rank': config.omp_threads_per_rank,
                'binding_strategy': config.binding_strategy,
                'placement_strategy': config.placement_strategy,
                'numa_policy': config.numa_policy,
                'description': config.description,
                'slurm_config': config.get_slurm_config(),
                'environment_variables': config.get_environment_variables(),
            }
            if config.num_nodes is not None:
                config_data["num_nodes"] = config.num_nodes
            configs_data.append(config_data)
        
        json_file = output_path / "generated_configurations.json"
        with open(json_file, 'w') as f:
            json.dump(configs_data, f, indent=2)

        # Export to CSV
        import pandas as pd
        csv_data = []
        
        for config in self.generated_configs:
            csv_data.append({
                'name': config.name,
                'mpi_ranks': config.mpi_ranks_per_node,
                'omp_threads': config.omp_threads_per_rank,
                'binding': config.binding_strategy,
                'placement': config.placement_strategy,
                'numa_policy': config.numa_policy,
                'description': config.description
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = output_path / "generated_configurations.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Exported configurations to {output_path}")
        logger.info(f"  - JSON: {json_file}")
        logger.info(f"  - CSV: {csv_file}")


def _parse_sinfo_cpus_field(token: str) -> Optional[int]:
    """
    Parse sinfo %c (CPUs per node) tokens such as '144', '72-72', '16-128+'.
    Returns a conservative count (minimum of any range) for layout generation.
    """
    if not token:
        return None
    s = token.strip().replace(" ", "")
    if not s:
        return None
    if "-" in s and not s.startswith("-"):
        parts = re.split(r"-+", s)
        nums: List[int] = []
        for p in parts:
            p = p.rstrip("+")
            if p.isdigit():
                nums.append(int(p))
        return min(nums) if nums else None
    s = s.rstrip("+")
    if s.isdigit():
        return int(s)
    return None


def query_slurm_cpus_per_node_cap(partition: str) -> Optional[int]:
    """Minimum CPUs/node reported by sinfo -N for this partition (safe for heterogeneous pools)."""
    if not partition or not str(partition).strip():
        return None
    try:
        r = subprocess.run(
            ["sinfo", "-h", "-p", str(partition).strip(), "-N", "-o", "%c"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        values: List[int] = []
        for line in r.stdout.splitlines():
            v = _parse_sinfo_cpus_field(line.strip())
            if v is not None and v > 0:
                values.append(v)
        return min(values) if values else None
    except Exception as e:
        logger.debug(f"sinfo CPU cap query failed for partition={partition!r}: {e}")
        return None


def query_scontrol_max_cpus_per_node(partition: str) -> Optional[int]:
    """Read MaxCPUsPerNode from `scontrol show partition` when the site defines it."""
    if not partition or not str(partition).strip():
        return None
    try:
        r = subprocess.run(
            ["scontrol", "show", "partition", str(partition).strip()],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0 or not r.stdout:
            return None
        m = re.search(r"\bMaxCPUsPerNode=(\d+)\b", r.stdout, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            return v if v > 0 else None
    except Exception as e:
        logger.debug(f"scontrol partition CPU cap failed for partition={partition!r}: {e}")
    return None


def _apply_slurm_cpu_layout_cap(
    physical_cores: int,
    sockets: int,
    numa_domains: int,
    architecture: str,
    slurm_partition: Optional[str],
) -> Tuple[int, str]:
    """
    Cap cores/node used for MPI×OpenMP layouts when SLURM allocates fewer than lscpu physical cores.

    Env:
      HPC_SLURM_CPUS_PER_NODE — explicit cap (applied with SLURM queries via min()).
    """
    import os

    caps: List[int] = []
    env_cap = os.environ.get("HPC_SLURM_CPUS_PER_NODE")
    if env_cap:
        try:
            v = int(str(env_cap).strip())
            if v > 0:
                caps.append(v)
        except ValueError:
            logger.warning(f"Ignoring invalid HPC_SLURM_CPUS_PER_NODE={env_cap!r}")

    part = (slurm_partition or os.environ.get("HPC_PARTITION") or "").strip() or None
    if part:
        q1 = query_slurm_cpus_per_node_cap(part)
        if q1:
            caps.append(q1)
        q2 = query_scontrol_max_cpus_per_node(part)
        if q2:
            caps.append(q2)

    if not caps:
        return physical_cores, ""

    cap = min(caps)
    if cap >= physical_cores:
        return physical_cores, ""

    logger.info(
        f"SLURM layout cap: using {cap} CPUs/node for configurations (lscpu/config had {physical_cores}); "
        f"partition={part!r}, caps tried={caps}"
    )
    return cap, f" ({cap} CPUs/node allocatable in SLURM)"


def detect_hardware_topology(
    system_name: str = None,
    slurm_partition: Optional[str] = None,
) -> HardwareTopology:
    """
    Detect hardware topology automatically using system commands.

    Uses lscpu (Linux) or sysctl (macOS). Optional SLURM partition is used to cap
    cores/node for layout generation when the scheduler exposes fewer CPUs than
    lscpu (common on some GPU partitions).

    Env:
      HPC_CORES_PER_NODE — force total physical cores (full override of lscpu).
      HPC_SLURM_CPUS_PER_NODE — max CPUs/node for layouts (with SLURM query caps).
      HPC_PARTITION — used when slurm_partition is None to query sinfo/scontrol.

    Args:
        system_name: Optional hint (logged only).
        slurm_partition: SLURM partition name for CPU cap queries (optional).

    Returns:
        HardwareTopology object with detected information
    """
    logger.info("Detecting hardware topology...")

    if system_name:
        logger.info(f"System hint provided: {system_name}")

    try:
        # Try to detect using system commands
        topology = _detect_from_system(slurm_partition=slurm_partition)
        if topology:
            logger.info(f"Detected hardware: {topology}")
            return topology
        
        # Fallback to generic defaults
        logger.warning("Could not detect hardware automatically, using generic defaults")
        return _get_default_topology()
        
    except Exception as e:
        logger.error(f"Error detecting hardware topology: {e}")
        return _get_default_topology()

def _detect_from_system(slurm_partition: Optional[str] = None) -> Optional[HardwareTopology]:
    """Detect hardware topology using system commands (Linux/Mac)"""
    try:
        import platform
        import os
        system_os = platform.system()
        
        # Linux detection using lscpu (works on HPC compute nodes)
        if system_os == 'Linux':
            try:
                lscpu_out = subprocess.check_output(['lscpu'], text=True, stderr=subprocess.DEVNULL)
                
                total_cores = 0
                sockets = 1
                numa_domains = 1
                threads_per_core = 1
                cores_per_socket_val = 0
                architecture = platform.machine()
                
                for line in lscpu_out.splitlines():
                    line_lower = line.lower()
                    if line_lower.startswith("cpu(s):"):
                        # This is total logical CPUs
                        total_cores = int(line.split(':')[1].strip())
                    elif "core(s) per socket:" in line_lower:
                        cores_per_socket_val = int(line.split(':')[1].strip())
                    elif "socket(s):" in line_lower:
                        sockets = int(line.split(':')[1].strip())
                    elif "thread(s) per core:" in line_lower:
                        threads_per_core = int(line.split(':')[1].strip())
                    elif "numa node(s):" in line_lower:
                        numa_domains = int(line.split(':')[1].strip())
                
                # Calculate physical cores (for HPC, we typically want physical cores, not hyperthreads)
                if cores_per_socket_val > 0 and sockets > 0:
                    physical_cores = cores_per_socket_val * sockets
                elif threads_per_core > 0:
                    physical_cores = total_cores // threads_per_core
                else:
                    physical_cores = total_cores
                
                if physical_cores > 0:
                    env_cores = os.environ.get("HPC_CORES_PER_NODE")
                    if env_cores:
                        physical_cores = int(env_cores)
                        cores_per_socket_val = physical_cores // max(1, sockets)

                    physical_cores, slurm_tag = _apply_slurm_cpu_layout_cap(
                        physical_cores,
                        sockets,
                        numa_domains,
                        architecture,
                        slurm_partition,
                    )
                    cores_per_socket_computed = physical_cores // max(1, sockets)

                    return HardwareTopology(
                        total_cores=physical_cores,
                        sockets=max(1, sockets),
                        cores_per_socket=cores_per_socket_computed,
                        numa_domains=max(1, numa_domains),
                        cores_per_numa=physical_cores // max(1, numa_domains),
                        memory_per_node_gb=HardwareTopology._estimate_memory_per_node(
                            physical_cores, architecture
                        ),
                        architecture=architecture,
                        system_name=(
                            f"HPC System ({physical_cores} cores, {numa_domains} NUMA)"
                            f"{slurm_tag}"
                        ),
                    )
            except Exception as e:
                logger.debug(f"Linux detection failed: {e}")

        # macOS detection using sysctl
        elif system_os == 'Darwin':
            try:
                total_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu'], text=True).strip())
                architecture = subprocess.check_output(['uname', '-m'], text=True).strip()
                mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True).strip())
                
                if total_cores > 0:
                    return HardwareTopology(
                        total_cores=total_cores,
                        sockets=1,
                        cores_per_socket=total_cores,
                        numa_domains=1,
                        cores_per_numa=total_cores,
                        memory_per_node_gb=mem_bytes / (1024**3),
                        architecture=architecture,
                        system_name=f"System ({total_cores} cores)"
                    )
            except Exception as e:
                logger.debug(f"macOS detection failed: {e}")
            
        return None
        
    except Exception as e:
        logger.warning(f"Could not detect hardware from system: {e}")
        return None

def _get_default_topology() -> HardwareTopology:
    """Get default hardware topology when detection fails"""
    # Generic default - 64 cores is a reasonable middle-ground for modern HPC nodes
    return HardwareTopology(
        total_cores=64,
        sockets=2,
        cores_per_socket=32,
        numa_domains=2,
        cores_per_numa=32,
        memory_per_node_gb=256.0,
        architecture="x86_64",
        system_name="HPC System (Default 64-core)"
    )

