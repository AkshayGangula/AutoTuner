#!/usr/bin/env python3
"""
LIKWID Integration for Memory Locality Metrics

This module provides integration with LIKWID (Like I Knew What I'm Doing) 
performance monitoring tool to extract hardware performance counter data
including NUMA locality and memory bandwidth metrics.

LIKWID provides access to:
- Memory bandwidth (local and remote)
- Cache miss rates
- NUMA efficiency
- CPU cycle counts
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LIKWIDMetrics:
    """Container for LIKWID-derived metrics"""
    # Memory bandwidth (GB/s)
    memory_read_bandwidth: float = 0.0
    memory_write_bandwidth: float = 0.0
    memory_bandwidth: float = 0.0
    
    # NUMA metrics
    local_dram_accesses: int = 0
    remote_dram_accesses: int = 0
    numa_efficiency: float = 0.0  # local/(local+remote)
    # Per-socket L3 load BW imbalance (first vs second half of HW threads); 0 = not computed
    socket_bandwidth_imbalance: float = 0.0
    
    # Cache metrics
    l2_cache_misses: int = 0
    l3_cache_misses: int = 0
    cache_miss_rate: float = 0.0
    
    # CPU metrics  
    cpu_cycles: int = 0
    instructions: int = 0
    ipc: float = 0.0  # Instructions per cycle
    
    # CPI stack (cycles per instruction breakdown)
    cpi_stack: Dict[str, float] = field(default_factory=dict)
    
    # Raw LIKWID output for debugging
    raw_output: str = ""


class LIKWIDProfiler:
    """
    LIKWID-based performance profiler for HPC systems
    
    Provides hardware counter profiling for memory access patterns and NUMA locality.
    """
    
    # LIKWID performance groups for different metrics
    PERFORMANCE_GROUPS = {
        'MEM': 'Memory bandwidth and data volume',
        'L2CACHE': 'L2 cache statistics',
        'L3CACHE': 'L3 cache statistics', 
        'NUMA': 'NUMA local vs remote memory access',
        'BRANCH': 'Branch prediction statistics',
        'FLOPS_DP': 'Double precision floating point operations',
        'ENERGY': 'Power and energy consumption',
    }
    
    # Recommended groups for auto-tuning
    # Priority: L3 (Core-local memory BW) > L2 > DATA. 
    # MEM/NUMA groups are REMOVED because they require root access (Uncore counters)
    # and fail on most shared HPC systems.
    DEFAULT_GROUPS = ['L3', 'L2', 'DATA']
    
    def __init__(self, work_directory: Path = None):
        """
        Initialize LIKWID profiler
        
        Args:
            work_directory: Directory for storing LIKWID output files
        """
        self.work_directory = work_directory or Path("likwid_output")
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self.likwid_available = self._check_likwid_available()
        
        if self.likwid_available:
            logger.info("LIKWID profiler initialized successfully")
        else:
            logger.warning("LIKWID not available on this system")
    
    def _check_likwid_available(self) -> bool:
        """Check if LIKWID is available on the system"""
        try:
            result = subprocess.run(
                ["likwid-perfctr", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip() if result.stdout else result.stderr.strip()
                logger.info(f"LIKWID version: {version}")
                return True
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_available_groups(self) -> List[str]:
        """Get list of available performance groups on this system"""
        if not self.likwid_available:
            return []
        
        try:
            result = subprocess.run(
                ["likwid-perfctr", "-a"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Some builds print the group table on stderr; "Group NAME" lines are valid.
            blob = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0 or blob.strip():
                groups = []
                for line in blob.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('-'):
                        continue
                    lower = line.lower()
                    if lower.startswith('group ') and len(line.split()) >= 2:
                        token = line.split()[1].strip('|:,')
                        if token:
                            groups.append(token)
                        continue
                    if lower == 'group' or lower.startswith('group name'):
                        continue
                    parts = line.split()
                    if parts:
                        groups.append(parts[0].strip('|:,'))
                return list(dict.fromkeys(groups))  # stable dedupe
            return []
        except Exception as e:
            logger.warning(f"Failed to get LIKWID groups: {e}")
            return []
    
    def generate_likwid_command(self, 
                                executable: str,
                                arguments: str = "",
                                groups: List[str] = None,
                                cpu_list: str = None,
                                output_file: Path = None) -> str:
        """
        Generate LIKWID command for performance monitoring
        
        Args:
            executable: Path to the executable to profile
            arguments: Command line arguments for the executable
            groups: List of LIKWID performance groups to measure
            cpu_list: CPU list specification (e.g., "0-47" or "0,12,24,36")
            output_file: Path to save LIKWID output (CSV format)
            
        Returns:
            Complete LIKWID command string
        """
        if groups is None:
            groups = self.DEFAULT_GROUPS
        
        # Build group specification (multiple groups with -g)
        group_spec = " ".join([f"-g {g}" for g in groups])
        
        # Build CPU specification
        cpu_spec = f"-C {cpu_list}" if cpu_list else ""
        
        # Build output specification
        output_spec = ""
        if output_file:
            output_spec = f"-o {output_file} -O"  # -O for CSV output
        
        # Construct full command
        cmd = f"likwid-perfctr {group_spec} {cpu_spec} {output_spec} {executable} {arguments}"
        
        return cmd.strip()
    
    def generate_slurm_likwid_commands(self, 
                                       executable: str,
                                       arguments: str = "",
                                       mpi_ranks: int = 1,
                                       omp_threads: int = 1,
                                       job_id: str = "default") -> List[str]:
        """
        Generate LIKWID commands for SLURM job scripts
        
        Args:
            executable: Path to the application executable
            arguments: Application arguments
            mpi_ranks: Number of MPI ranks
            omp_threads: Number of OpenMP threads per rank
            job_id: Job identifier for output files
            
        Returns:
            List of shell commands to add to SLURM script
        """
        output_dir = self.work_directory / job_id
        output_file = output_dir / "likwid_output.csv"
        
        commands = [
            f"# LIKWID Performance Monitoring",
            f"export LIKWID_OUTPUT_DIR={output_dir}",
            f"mkdir -p $LIKWID_OUTPUT_DIR",
            "",
        ]
        
        if mpi_ranks == 1:
            # Single rank: Direct LIKWID invocation
            likwid_cmd = self.generate_likwid_command(
                executable=executable,
                arguments=arguments,
                groups=['MEM', 'L3CACHE'],
                cpu_list=f"0-{omp_threads-1}",
                output_file=output_file
            )
            commands.append(f"# Single-rank LIKWID profiling")
            commands.append(likwid_cmd)
        else:
            # Multi-rank: Use likwid-mpirun wrapper or per-rank profiling
            # Option 1: Use likwid-mpirun (if available)
            # Option 2: Profile rank 0 only with LIKWID
            commands.append(f"# Multi-rank: Profile rank 0 with LIKWID")
            commands.append(f"if [ $SLURM_PROCID -eq 0 ]; then")
            
            # Calculate CPU cores for this rank
            start_cpu = 0
            end_cpu = omp_threads - 1
            
            likwid_cmd = self.generate_likwid_command(
                executable=executable,
                arguments=arguments,
                groups=['MEM', 'L3CACHE'],
                cpu_list=f"{start_cpu}-{end_cpu}",
                output_file=output_file
            )
            commands.append(f"    {likwid_cmd}")
            commands.append(f"else")
            commands.append(f"    {executable} {arguments}")
            commands.append(f"fi")
        
        return commands
    
    def parse_likwid_output(self, output_file: Path) -> LIKWIDMetrics:
        """
        Parse LIKWID output file (CSV or text format)
        
        Args:
            output_file: Path to LIKWID output file
            
        Returns:
            LIKWIDMetrics object with extracted metrics
        """
        metrics = LIKWIDMetrics()
        
        if not output_file.exists():
            logger.warning(f"LIKWID output file not found: {output_file}")
            return metrics
        
        try:
            content = output_file.read_text()
            metrics.raw_output = content
            
            # Parse based on file extension
            if output_file.suffix == '.csv':
                metrics = self._parse_csv_output(content, metrics)
            else:
                metrics = self._parse_text_output(content, metrics)
            
            # Calculate derived metrics
            metrics = self._calculate_derived_metrics(metrics)
            
            logger.info(f"Parsed LIKWID metrics: NUMA efficiency={metrics.numa_efficiency:.3f}, "
                       f"Memory BW={metrics.memory_bandwidth:.2f} GB/s")
            
        except Exception as e:
            logger.error(f"Failed to parse LIKWID output: {e}")
        
        return metrics
    
    def _parse_csv_output(self, content: str, metrics: LIKWIDMetrics) -> LIKWIDMetrics:
        """Parse LIKWID CSV format output (table format with Event,Counter,HWThread columns or Metric rows)."""
        lines = content.strip().split('\n')
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            
            metric_name = parts[0].strip().lower()
            
            # Extract numeric values from line (skipping name and counter/unit columns)
            numeric_values = []
            for i in range(1, len(parts)):
                try:
                    val = float(parts[i])
                    numeric_values.append(val)
                except ValueError:
                    pass
            
            if not numeric_values:
                continue

            # Determine value to use
            line_value = 0.0
            is_stat = "stat" in metric_name
            
            if is_stat:
                # STAT rows: Sum, Min, Max, Avg. For bandwidth, Sum is the aggregate socket rate (MBytes/s).
                if 'bandwidth' in metric_name and len(numeric_values) >= 1:
                    line_value = numeric_values[0]
                else:
                    line_value = numeric_values[0]
            else:
                # Per-HWThread metric row: sum is wrong for bandwidth (double-counts); average like CPI/Clock
                if any(x in metric_name for x in ['clock', 'cpi', 'ipc', 'rate', 'freq', 'bandwidth']):
                    line_value = sum(numeric_values) / max(len(numeric_values), 1)
                else:
                    line_value = sum(numeric_values)
            
            # Map to metrics attributes
            target_attr = None
            is_integer = False
            is_bandwidth = False
            
            if 'memory read bandwidth' in metric_name or 'memory bw read' in metric_name:
                target_attr = 'memory_read_bandwidth'
                is_bandwidth = True
            elif 'memory write bandwidth' in metric_name or 'memory bw write' in metric_name:
                target_attr = 'memory_write_bandwidth'
                is_bandwidth = True
            elif 'memory bandwidth' in metric_name or 'l2<->l3 bandwidth' in metric_name:
                target_attr = 'memory_bandwidth'
                is_bandwidth = True
            # Ice Lake / many Intel paths: Run 1b uses L3 group — total "L3 bandwidth" is the useful DRAM+L3 estimate
            elif (
                'l3 bandwidth' in metric_name
                and 'l3 load' not in metric_name
                and 'l3 evict' not in metric_name
                and 'l3|mem' not in metric_name
                and 'dropped' not in metric_name
            ):
                target_attr = 'memory_bandwidth'
                is_bandwidth = True
            # AMD NUMA group: "Local data volume [GBytes]" / "Remote data volume [GBytes]"
            # Use data volume to estimate local/remote DRAM accesses (1 GB = ~15.6M cache lines)
            elif 'local data volume' in metric_name and 'remote' not in metric_name:
                target_attr = 'local_dram_accesses'
                is_integer = True
                # Convert GBytes to cache line count (64-byte lines): GB * 1e9 / 64 ≈ GB * 15.6M
                line_value = int(line_value * 15625000) if line_value > 0 else 0
            elif 'remote data volume' in metric_name:
                target_attr = 'remote_dram_accesses'
                is_integer = True
                # Convert GBytes to cache line count
                line_value = int(line_value * 15625000) if line_value > 0 else 0
            elif ('local dram' in metric_name or 'mem_load_l3_miss_retired_local_dram' in metric_name) and 'remote' not in metric_name:
                target_attr = 'local_dram_accesses'
                is_integer = True
            elif 'remote dram' in metric_name or 'mem_load_l3_miss_retired_remote_dram' in metric_name:
                target_attr = 'remote_dram_accesses'
                is_integer = True
            elif 'l2 miss' in metric_name:
                target_attr = 'l2_cache_misses'
                is_integer = True
            elif 'l3 miss' in metric_name:
                target_attr = 'l3_cache_misses'
                is_integer = True
            elif 'l2d_cache_refill' in metric_name:
                target_attr = 'l3_cache_misses'
                is_integer = True
            elif 'inst_retired' in metric_name:
                target_attr = 'instructions'
                is_integer = True
            elif 'ipc' in metric_name or 'instructions per cycle' in metric_name:
                target_attr = 'ipc'
            elif 'cpu cycles' in metric_name:
                target_attr = 'cpu_cycles'
                is_integer = True
            elif 'instructions' in metric_name:
                target_attr = 'instructions'
                is_integer = True
                
            if target_attr:
                # Apply unit conversion for bandwidth (MBytes/s -> GB/s)
                # Always convert MBytes/s to GB/s regardless of magnitude
                if is_bandwidth:
                     line_value = line_value / 1000.0
                
                # Update if new value is non-zero (CRITICAL FIX: prevents zero-valued STAT lines from overwriting valid raw sums)
                if line_value > 0:
                    if is_integer:
                        setattr(metrics, target_attr, int(line_value))
                    else:
                        setattr(metrics, target_attr, float(line_value))

            # Socket traffic balance: compare mean L3 *load* BW on first vs second half of HW threads (2-socket heuristic).
            if (
                not is_stat
                and "l3 load bandwidth" in metric_name
                and len(numeric_values) >= 8
            ):
                half = len(numeric_values) // 2
                if half >= 4:
                    s0 = sum(numeric_values[:half]) / half
                    s1 = sum(numeric_values[half:]) / max(len(numeric_values) - half, 1)
                    denom = s0 + s1
                    if denom > 1e-9:
                        imb = abs(s0 - s1) / denom
                        metrics.socket_bandwidth_imbalance = max(
                            metrics.socket_bandwidth_imbalance, float(imb)
                        )
        
        return metrics
    
    def _parse_text_output(self, content: str, metrics: LIKWIDMetrics) -> LIKWIDMetrics:
        """Parse LIKWID standard text format output"""
        
        # First, try to parse STAT summary lines (aggregated values) - these are more reliable
        # Look for patterns like "Memory bandwidth [MBytes/s] STAT | 0 | 0 | 0 | 0"
        # The Sum column is typically the 2nd value after the pipe
        stat_patterns = {
            'memory_read_bandwidth': r'Memory read bandwidth\s*\[MBytes/s\]\s*STAT.*?\|\s*[\d.]+\s*\|\s*([\d.]+)',
            'memory_write_bandwidth': r'Memory write bandwidth\s*\[MBytes/s\]\s*STAT.*?\|\s*[\d.]+\s*\|\s*([\d.]+)',
            'memory_bandwidth': r'(?:Memory|L3) bandwidth\s*\[MBytes/s\]\s*STAT.*?\|\s*[\d.]+\s*\|\s*([\d.]+)',
            'local_dram': r'Local DRAM\s*STAT.*?\|\s*\d+\s*\|\s*(\d+)',
            'remote_dram': r'Remote DRAM\s*STAT.*?\|\s*\d+\s*\|\s*(\d+)',
        }
        
        # Try STAT patterns first (more reliable - uses Sum column)
        for metric_name, pattern in stat_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1)
                try:
                    if metric_name == 'memory_read_bandwidth':
                        metrics.memory_read_bandwidth = float(value) / 1000  # Convert to GB/s
                    elif metric_name == 'memory_write_bandwidth':
                        metrics.memory_write_bandwidth = float(value) / 1000
                    elif metric_name == 'memory_bandwidth':
                        metrics.memory_bandwidth = float(value) / 1000
                    elif metric_name == 'local_dram':
                        metrics.local_dram_accesses = int(value)
                    elif metric_name == 'remote_dram':
                        metrics.remote_dram_accesses = int(value)
                except (ValueError, TypeError):
                    pass
        
        # Fallback 2: ASCII Table format lines (e.g. | L3 bandwidth [MBytes/s] | 123.45 |)
        # Matches the first value column (HWThread 0)
        table_patterns = {
             'memory_read_bandwidth': r'\|\s*Memory read bandwidth\s*\[MBytes/s\]\s*\|\s*([\d.]+)',
             'memory_write_bandwidth': r'\|\s*Memory write bandwidth\s*\[MBytes/s\]\s*\|\s*([\d.]+)',
             'memory_bandwidth': r'\|\s*(?:Memory|L3) bandwidth\s*\[MBytes/s\]\s*\|\s*([\d.]+)',
             'local_dram': r'\|\s*Local DRAM\s*\|\s*(\d+)',
             'remote_dram': r'\|\s*Remote DRAM\s*\|\s*(\d+)',
             'l2_miss_rate': r'\|\s*L2 miss rate\s*\|\s*([\d.]+)',
             'l3_miss_rate': r'\|\s*L3 miss rate\s*\|\s*([\d.]+)',
             'ipc': r'\|\s*IPC\s*\|\s*([\d.]+)',
             'cpu_cycles': r'\|\s*CPU cycles\s*\|\s*(\d+)',
             'instructions': r'\|\s*Instructions\s*\|\s*(\d+)',
        }

        for metric_name, pattern in table_patterns.items():
             # Skip if we already found this metric
            if metric_name == 'memory_read_bandwidth' and metrics.memory_read_bandwidth > 0: continue
            if metric_name == 'memory_write_bandwidth' and metrics.memory_write_bandwidth > 0: continue
            if metric_name == 'memory_bandwidth' and metrics.memory_bandwidth > 0: continue
            
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    if metric_name == 'memory_bandwidth':
                         metrics.memory_bandwidth = float(value) / 1000
                    elif metric_name == 'l3_miss_rate':
                         metrics.cache_miss_rate = float(value)
                    elif metric_name == 'cpu_cycles':
                         metrics.cpu_cycles = int(value)
                    elif metric_name == 'instructions':
                         metrics.instructions = int(value)
                    # Add other mapping logic as needed (similar to above)
                except (ValueError, TypeError):
                    pass

        # Fallback 3: Regular expressions for Key: Value format
        patterns = {
            'memory_read_bandwidth': r'Memory read bandwidth\s*\[MBytes/s\]\s*:\s*([\d.]+)',
            'memory_write_bandwidth': r'Memory write bandwidth\s*\[MBytes/s\]\s*:\s*([\d.]+)',
            'memory_bandwidth': r'(?:Memory|L3) bandwidth\s*\[MBytes/s\]\s*:\s*([\d.]+)',
            'local_dram': r'Local DRAM\s*:\s*(\d+)',
            'remote_dram': r'Remote DRAM\s*:\s*(\d+)',
            'l2_miss_rate': r'L2 miss rate\s*:\s*([\d.]+)',
            'l3_miss_rate': r'L3 miss rate\s*:\s*([\d.]+)',
            'ipc': r'IPC\s*:\s*([\d.]+)',
            'cpu_cycles': r'CPU cycles\s*:\s*(\d+)',
            'instructions': r'Instructions\s*:\s*(\d+)',
        }
        
        # Only use fallback if STAT patterns didn't find values
        for metric_name, pattern in patterns.items():
            # Skip if we already found this metric from STAT
            if metric_name == 'memory_read_bandwidth' and metrics.memory_read_bandwidth > 0:
                continue
            if metric_name == 'memory_write_bandwidth' and metrics.memory_write_bandwidth > 0:
                continue
            if metric_name == 'memory_bandwidth' and metrics.memory_bandwidth > 0:
                continue
            if metric_name == 'local_dram' and metrics.local_dram_accesses > 0:
                continue
            if metric_name == 'remote_dram' and metrics.remote_dram_accesses > 0:
                continue
            
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    if metric_name == 'memory_read_bandwidth':
                        metrics.memory_read_bandwidth = float(value) / 1000  # Convert to GB/s
                    elif metric_name == 'memory_write_bandwidth':
                        metrics.memory_write_bandwidth = float(value) / 1000
                    elif metric_name == 'memory_bandwidth':
                        metrics.memory_bandwidth = float(value) / 1000
                    elif metric_name == 'local_dram':
                        metrics.local_dram_accesses = int(value)
                    elif metric_name == 'remote_dram':
                        metrics.remote_dram_accesses = int(value)
                    elif metric_name == 'l2_miss_rate':
                        metrics.l2_cache_misses = int(float(value))
                    elif metric_name == 'l3_miss_rate':
                        metrics.cache_miss_rate = float(value)
                    elif metric_name == 'ipc':
                        metrics.ipc = float(value)
                    elif metric_name == 'cpu_cycles':
                        metrics.cpu_cycles = int(value)
                    elif metric_name == 'instructions':
                        metrics.instructions = int(value)
                except (ValueError, TypeError):
                    pass
        
        return metrics
    
    def _calculate_derived_metrics(self, metrics: LIKWIDMetrics) -> LIKWIDMetrics:
        """Calculate derived metrics from raw LIKWID data. Validates against garbage values."""
        
        # Validation: Check for impossible values (likely error codes from locked counters)
        MAX_VALID_BANDWIDTH = 20000.0  # 20 TB/s (Generous limit)
        # 2^47 is approx 1.4e14. Error codes often appear as absolute max values.
        MAX_VALID_ACCESSES = 1e14
        
        if metrics.memory_bandwidth > MAX_VALID_BANDWIDTH or \
           metrics.memory_read_bandwidth > MAX_VALID_BANDWIDTH or \
           metrics.memory_write_bandwidth > MAX_VALID_BANDWIDTH:
            logger.warning(f"LIKWID reported impossible bandwidth ({metrics.memory_bandwidth:.2f} GB/s). Treating as invalid (0.0).")
            metrics.memory_bandwidth = 0.0
            metrics.memory_read_bandwidth = 0.0
            metrics.memory_write_bandwidth = 0.0
            
        if metrics.local_dram_accesses > MAX_VALID_ACCESSES or \
           metrics.remote_dram_accesses > MAX_VALID_ACCESSES:
            logger.warning(f"LIKWID reported impossible DRAM accesses ({metrics.local_dram_accesses}). Treating as invalid.")
            metrics.local_dram_accesses = 0
            metrics.remote_dram_accesses = 0

        # NUMA efficiency: genuine only when we have local/remote DRAM counts from LIKWID
        total_dram = metrics.local_dram_accesses + metrics.remote_dram_accesses
        if total_dram > 0:
            metrics.numa_efficiency = metrics.local_dram_accesses / total_dram
        else:
            # 0.0 means "not measured" (empty/unparsed LIKWID output)
            metrics.numa_efficiency = 0.0
        
        if metrics.memory_bandwidth == 0:
            metrics.memory_bandwidth = metrics.memory_read_bandwidth + metrics.memory_write_bandwidth

        # Grace Hopper Locality extraction from L2 refills (mapped to l3_cache_misses) and instructions
        if metrics.cache_miss_rate == 0.0 and metrics.instructions > 0 and getattr(metrics, 'l3_cache_misses', 0) > 0:
            # Fraction of instructions that resulted in an L2 refill
            refill_rate = metrics.l3_cache_misses / metrics.instructions
            # Scale it to a plausible 0.0-1.0 miss rate (empirical factor for Grace cache hierarchy)
            metrics.cache_miss_rate = min(1.0, refill_rate * 100.0)

        # No DRAM counters: socket L3-load balance × L3 BW estimate (heuristic; not true local/remote DRAM)
        if total_dram == 0 and metrics.numa_efficiency == 0.0:
            bal_eff = 0.0
            if metrics.socket_bandwidth_imbalance > 0:
                bal_eff = max(0.0, min(1.0, 1.0 - metrics.socket_bandwidth_imbalance))
            bw_eff = min(1.0, metrics.memory_bandwidth / 80.0) if metrics.memory_bandwidth > 0 else 0.0
            if bal_eff > 0 and bw_eff > 0:
                metrics.numa_efficiency = 0.55 * bal_eff + 0.45 * bw_eff
            elif bal_eff > 0:
                metrics.numa_efficiency = bal_eff
            elif bw_eff > 0:
                metrics.numa_efficiency = bw_eff

        # Fallback for Grace Hopper: Map simulated hit rate to genuine numa_efficiency so it's scored properly
        if metrics.numa_efficiency == 0.0 and metrics.cache_miss_rate > 0:
            metrics.numa_efficiency = 1.0 - metrics.cache_miss_rate
        
        # Calculate IPC if not provided but we have cycles and instructions
        if metrics.ipc == 0 and metrics.cpu_cycles > 0 and metrics.instructions > 0:
            metrics.ipc = metrics.instructions / metrics.cpu_cycles
        
        return metrics
    
    def extract_metrics_for_job(self, job_id: str) -> Optional[LIKWIDMetrics]:
        """
        Extract LIKWID metrics for a completed job
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            LIKWIDMetrics object or None if not available
        """
        # Look for LIKWID output files in various locations
        possible_paths = [
            self.work_directory / job_id / "likwid_output.csv",
            self.work_directory / job_id / "likwid_output.txt",
            Path(f"likwid_output/{job_id}/likwid_output.csv"),
            Path(f"likwid_output/{job_id}/likwid_output.txt"),
        ]
        
        for output_path in possible_paths:
            if output_path.exists():
                logger.info(f"Found LIKWID output: {output_path}")
                return self.parse_likwid_output(output_path)
        
        logger.debug(f"No LIKWID output found for job {job_id}")
        return None
    
    def get_metrics_dict(self, metrics: LIKWIDMetrics) -> Dict[str, Any]:
        """
        Convert LIKWIDMetrics to dictionary for integration with auto-tuner
        
        Args:
            metrics: LIKWIDMetrics object
            
        Returns:
            Dictionary with metrics ready for scoring
        """
        return {
            'memory_bandwidth': metrics.memory_bandwidth,
            'memory_read_bandwidth': metrics.memory_read_bandwidth,
            'memory_write_bandwidth': metrics.memory_write_bandwidth,
            'numa_efficiency': metrics.numa_efficiency,
            'local_dram_accesses': metrics.local_dram_accesses,
            'remote_dram_accesses': metrics.remote_dram_accesses,
            'cache_miss_rate': metrics.cache_miss_rate,
            'l2_cache_misses': metrics.l2_cache_misses,
            'l3_cache_misses': metrics.l3_cache_misses,
            'ipc': metrics.ipc,
            'cpu_cycles': metrics.cpu_cycles,
            'instructions': metrics.instructions,
            'socket_bandwidth_imbalance': metrics.socket_bandwidth_imbalance,
        }


def resolve_likwid_output_dir(experiment_dir: Path, job_id: str) -> Optional[Path]:
    """
    LIKWID CSVs may live under <experiment>/likwid_output/<job_id> (compute writes here)
    or only under <experiment>/results/<job_id>/likwid_output (after collection).
    """
    jid = str(job_id)
    for d in (
        experiment_dir / "likwid_output" / jid,
        experiment_dir / "results" / jid / "likwid_output",
    ):
        if d.is_dir() and (
            any(d.glob("likwid*.csv")) or any(d.glob("likwid*.txt"))
        ):
            return d
    return None


def parse_numa_meminfo(experiment_dir: Path, job_id: str) -> tuple:
    """
    Parse the kernel NUMA page-allocation snapshot written by the SLURM script.

    The file ``$LIKWID_OUTPUT_DIR/numa_meminfo.txt`` starts with a SOURCE line
    indicating which sysfs file was read, then BEFORE/AFTER snapshots.

    Two file formats are handled:

    **numastat** format (``SOURCE numastat``):
        Per-node file ``/sys/devices/system/node/nodeN/numastat`` — always
        present on NUMA-capable kernels. Fields are lowercase, no ``Node N``
        prefix, and the file is repeated once per NUMA node::

            local_node 12345678
            other_node 12345
            numa_hit   12345678
            numa_miss  12345

        ``local_node``/``numa_hit`` = allocations served locally.
        ``other_node``/``numa_miss`` = allocations fetched from remote nodes.

    **meminfo** format (``SOURCE meminfo``):
        Fields inside ``/sys/devices/system/node/nodeN/meminfo`` when the
        kernel is compiled with ``CONFIG_NUMA_STAT=y``::

            Node 0 LocalNode:  12345678
            Node 0 OtherNode:  12345

    Returns:
        (local_pages, remote_pages, numa_locality_fraction)
        All three are 0 / 0.0 if the file is missing or unparseable.
    """
    d = resolve_likwid_output_dir(experiment_dir, job_id)
    if d is None:
        return 0, 0, 0.0

    meminfo_file = d / "numa_meminfo.txt"
    if not meminfo_file.is_file():
        return 0, 0, 0.0

    import re
    # meminfo-style:  "Node 0 LocalNode:  123"  or  "Node 0 NumaHit:  123"
    meminfo_local_re = re.compile(r'Node\s+\d+\s+(?:LocalNode|NumaHit):\s+(\d+)', re.IGNORECASE)
    meminfo_other_re = re.compile(r'Node\s+\d+\s+(?:OtherNode|NumaMiss):\s+(\d+)', re.IGNORECASE)
    # numastat-style: "local_node 123"  at start of line (preferred; avoids double-count with numa_hit)
    numastat_local_re = re.compile(r'^local_node\s+(\d+)', re.IGNORECASE)
    numastat_other_re = re.compile(r'^other_node\s+(\d+)', re.IGNORECASE)

    before_local: list = []
    before_other: list = []
    after_local:  list = []
    after_other:  list = []

    source = "meminfo"   # default; overridden by SOURCE header
    section = None
    try:
        for line in meminfo_file.read_text().splitlines():
            line = line.strip()
            if line == "SOURCE":
                section = "_source_next"
                continue
            if section == "_source_next":
                source = line.lower()
                section = None
                continue
            if line == "BEFORE":
                section = "before"
                continue
            elif line == "AFTER":
                section = "after"
                continue
            if section not in ("before", "after"):
                continue

            dest_local = before_local if section == "before" else after_local
            dest_other = before_other if section == "before" else after_other

            if source == "numastat":
                m = numastat_local_re.match(line)
                if m:
                    dest_local.append(int(m.group(1)))
                    continue
                m = numastat_other_re.match(line)
                if m:
                    dest_other.append(int(m.group(1)))
            else:
                m = meminfo_local_re.search(line)
                if m:
                    dest_local.append(int(m.group(1)))
                    continue
                m = meminfo_other_re.search(line)
                if m:
                    dest_other.append(int(m.group(1)))
    except Exception:
        return 0, 0, 0.0

    if not after_local or not after_other:
        return 0, 0, 0.0

    n = min(len(before_local), len(after_local), len(before_other), len(after_other))
    if n == 0:
        return 0, 0, 0.0

    delta_local = max(0, sum(after_local[i] - before_local[i] for i in range(n)))
    delta_other = max(0, sum(after_other[i] - before_other[i] for i in range(n)))

    total = delta_local + delta_other
    if total == 0:
        return 0, 0, 0.0

    return delta_local, delta_other, delta_local / total


def likwid_aggregate_memory_bandwidth_gbps(experiment_dir: Path, job_id: str) -> float:
    """Best memory bandwidth (GB/s) from likwid_*.csv under the job's LIKWID directory (max per file)."""
    d = resolve_likwid_output_dir(experiment_dir, job_id)
    if d is None:
        return 0.0
    try:
        prof = LIKWIDProfiler()
        best = 0.0
        for f in sorted(d.glob("likwid_*.csv")):
            m = prof.parse_likwid_output(f)
            best = max(best, float(m.memory_bandwidth or 0.0))
        return best
    except Exception:
        return 0.0


def resolve_memory_bandwidth_for_job(
    experiment_dir: Path, job_id: str, nsight_bandwidth_gbps: float = 0.0
) -> Tuple[float, str]:
    """
    Pick γ memory-bandwidth input: prefer LIKWID L3/MEM when Nsight only has a tiny CPU-derived value.
    Returns (gbps, source_tag).
    """
    nsight_bw = float(nsight_bandwidth_gbps or 0.0)
    likwid_bw = likwid_aggregate_memory_bandwidth_gbps(experiment_dir, job_id)
    noise_floor = 0.05
    if likwid_bw > max(nsight_bw, noise_floor):
        return likwid_bw, "likwid_l3_or_mem"
    if likwid_bw > nsight_bw > 0:
        return likwid_bw, "likwid_l3_or_mem"
    if nsight_bw > 0:
        return nsight_bw, "nsight_derived"
    if likwid_bw > 0:
        return likwid_bw, "likwid_l3_or_mem"
    return 0.0, "missing"


def get_numa_efficiency_for_job(experiment_dir: Path, job_id: str) -> float:
    """
    Read LIKWID output for a job (single or multi-rank) and return aggregated NUMA efficiency.
    Genuine only: uses local/(local+remote) DRAM from LIKWID. Returns 0.0 when no local/remote
    data (empty or unparsed output) — no fallback.
    """
    likwid_dir = resolve_likwid_output_dir(experiment_dir, job_id)
    if likwid_dir is None:
        return 0.0
    try:
        profiler = LIKWIDProfiler()
        files = list(likwid_dir.glob("likwid_output.csv")) + list(likwid_dir.glob("likwid_output.txt"))
        files += sorted(likwid_dir.glob("likwid_*.csv")) + sorted(likwid_dir.glob("likwid_*.txt"))
        files = list(dict.fromkeys(files))
        if not files:
            return 0.0
        all_local, all_remote = 0, 0
        total_miss_rate, count = 0.0, 0
        total_likwid_bw = 0.0
        total_imbalance, imb_count = 0.0, 0
        for f in files:
            m = profiler.parse_likwid_output(f)
            all_local += m.local_dram_accesses
            all_remote += m.remote_dram_accesses
            total_likwid_bw += float(m.memory_bandwidth or 0.0)
            if m.cache_miss_rate > 0:
                total_miss_rate += m.cache_miss_rate
                count += 1
            if m.socket_bandwidth_imbalance > 0:
                total_imbalance += m.socket_bandwidth_imbalance
                imb_count += 1
                
        total = all_local + all_remote
        if total > 0:
            return all_local / total
            
        # Fallback: Use L3 cache hit rate for locality on Grace Hopper when DRAM counters unavailable
        if count > 0:
            avg_miss_rate = total_miss_rate / count
            # Ensure miss rate is a fraction. If it's a percentage (e.g., > 1.0), convert it
            if avg_miss_rate > 1.0:
                avg_miss_rate /= 100.0
            efficiency = max(0.0, 1.0 - avg_miss_rate)
            logger.info(f"LIKWID: Using L3 Cache Hit Rate ({efficiency:.3f}) as fallback for locality when DRAM not available.")
            return efficiency

        ref = 80.0
        avg_imb = total_imbalance / imb_count if imb_count else 0.0
        balance_eff = max(0.0, 1.0 - avg_imb) if avg_imb > 0 else 0.0
        bw_eff = min(1.0, total_likwid_bw / ref) if total_likwid_bw > 0 else 0.0
        if balance_eff > 0 and bw_eff > 0:
            eff = 0.55 * balance_eff + 0.45 * bw_eff
            logger.info(
                f"LIKWID: socket L3 balance + BW estimate ({eff:.3f}) for locality (no DRAM counters)."
            )
            return eff
        if balance_eff > 0:
            logger.info(f"LIKWID: socket L3 balance estimate ({balance_eff:.3f}) for locality.")
            return balance_eff

        if total_likwid_bw > 0:
            eff = bw_eff
            logger.info(f"LIKWID: Using L3 bandwidth estimate ({eff:.3f}) for locality (no NUMA/MBOX DRAM counters).")
            return eff
            
        if files:
            logger.debug("LIKWID: no local/remote DRAM or L3 miss rate in %d file(s) — locality 0.0 (not measured)", len(files))
        return 0.0
    except Exception:
        return 0.0


def get_locality_for_job(experiment_dir: Path, job_id: str) -> tuple[float, str]:
    """
    Return (locality_efficiency, source) for a job's LIKWID outputs.

    source:
    - "dram_local_remote": genuine local/(local+remote) from LIKWID
    - "l3_estimate": fallback from cache-miss-derived L3 hit rate
    - "socket_l3_balance_bw_estimate": blend of per-socket L3 load balance + L3 BW estimate
    - "socket_l3_balance_estimate": socket L3 load balance only
    - "l3_bandwidth_estimate": L3 (or memory) bandwidth normalization only
    - "missing": no usable LIKWID locality signal
    """
    likwid_dir = resolve_likwid_output_dir(experiment_dir, job_id)
    if likwid_dir is None:
        return 0.0, "missing"
    try:
        profiler = LIKWIDProfiler()
        files = list(likwid_dir.glob("likwid_output.csv")) + list(likwid_dir.glob("likwid_output.txt"))
        files += sorted(likwid_dir.glob("likwid_*.csv")) + sorted(likwid_dir.glob("likwid_*.txt"))
        files = list(dict.fromkeys(files))
        if not files:
            return 0.0, "missing"

        all_local, all_remote = 0, 0
        total_miss_rate, count = 0.0, 0
        total_likwid_bw = 0.0
        total_imbalance, imb_count = 0.0, 0
        for f in files:
            m = profiler.parse_likwid_output(f)
            all_local += m.local_dram_accesses
            all_remote += m.remote_dram_accesses
            total_likwid_bw += float(m.memory_bandwidth or 0.0)
            if m.cache_miss_rate > 0:
                total_miss_rate += m.cache_miss_rate
                count += 1
            if m.socket_bandwidth_imbalance > 0:
                total_imbalance += m.socket_bandwidth_imbalance
                imb_count += 1

        total = all_local + all_remote
        if total > 0:
            return (all_local / total), "dram_local_remote"

        if count > 0:
            avg_miss_rate = total_miss_rate / count
            if avg_miss_rate > 1.0:
                avg_miss_rate /= 100.0
            efficiency = max(0.0, 1.0 - avg_miss_rate)
            return efficiency, "l3_estimate"

        ref = 80.0
        avg_imb = total_imbalance / imb_count if imb_count else 0.0
        balance_eff = max(0.0, 1.0 - avg_imb) if avg_imb > 0 else 0.0
        bw_eff = min(1.0, total_likwid_bw / ref) if total_likwid_bw > 0 else 0.0
        if balance_eff > 0 and bw_eff > 0:
            return 0.55 * balance_eff + 0.45 * bw_eff, "socket_l3_balance_bw_estimate"
        if balance_eff > 0:
            return balance_eff, "socket_l3_balance_estimate"
        if total_likwid_bw > 0:
            return bw_eff, "l3_bandwidth_estimate"
        return 0.0, "missing"
    except Exception:
        return 0.0, "missing"


def check_likwid_on_system() -> Dict[str, Any]:
    """
    Check LIKWID availability and capabilities on current system
    
    Returns:
        Dictionary with LIKWID status and available features
    """
    status = {
        'available': False,
        'version': None,
        'groups': [],
        'error': None
    }
    
    try:
        # Check if likwid-perfctr is available
        result = subprocess.run(
            ["likwid-perfctr", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            status['available'] = True
            status['version'] = result.stdout.strip() or result.stderr.strip()
            profiler = LIKWIDProfiler()
            status['groups'] = profiler.get_available_groups()
        else:
            status['error'] = result.stderr.strip() if result.stderr else "Unknown error"
            
    except FileNotFoundError:
        status['error'] = "LIKWID not found. Try: module load likwid"
    except subprocess.TimeoutExpired:
        status['error'] = "LIKWID command timed out"
    except Exception as e:
        status['error'] = str(e)
    
    return status


