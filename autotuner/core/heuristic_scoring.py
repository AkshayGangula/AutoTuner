#!/usr/bin/env python3
"""
Complete Heuristic Scoring System for MPI+OpenMP Auto-Tuning

Implements the enhanced 5-dimensional composite scoring function using complete dataset analysis:
Score_conf = α(1-MPI_comm) + β(1-Thread_stall) + γ Locality_eff + δ GPU_util + ε OpenMP_eff

Utilizes ALL tuples from ALL tables for comprehensive performance analysis:
- Complete MPI communication analysis from OSRT_API events
- Complete thread performance analysis from SCHED_EVENTS and COMPOSITE_EVENTS
- Complete GPU performance analysis from CUPTI tables
- Complete memory locality analysis from all memory-related tables
- Complete ENUM mappings for accurate event classification

Where:
- MPI_comm: Normalized MPI communication volume using complete OSRT_API analysis
- Thread_stall: Proportion of thread idle time using complete scheduling analysis
- Locality_eff: Memory access locality efficiency using complete memory analysis
- GPU_util: GPU utilization efficiency using complete CUDA analysis
- OpenMP_eff: OpenMP work distribution efficiency using complete OpenMP analysis
- α, β, γ, δ, ε: Tunable weights (default: 0.3, 0.3, 0.2, 0.1, 0.1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics using complete dataset analysis"""
    mpi_comm_time: float = 0.0
    mpip_app_time: float = 0.0
    mpip_max_mpi_wall_fraction: float = 0.0
    total_runtime: float = 0.0
    cpu_utilization: float = 0.0
    memory_local_accesses: int = 0
    memory_total_accesses: int = 0
    numa_efficiency: float = 0.0
    thread_stall_time: float = 0.0
    load_imbalance: float = 0.0
    cache_miss_rate: float = 0.0
    
    # Enhanced metrics
    gpu_utilization: float = 0.0
    gpu_memory_bandwidth: float = 0.0
    openmp_work_efficiency: float = 0.0
    # False when metrics come from stdout/LIKWID/mpiP only (no Nsight SQLite)
    profiling_trace_present: bool = True
    memory_transfer_rate: float = 0.0
    memory_bottleneck_severity: float = 0.0
    throughput_gflops: float = 0.0
    
    # Application metadata
    application_type: str = "UNKNOWN"
    problem_size: int = 0
    iterations: int = 0
    
    # LIKWID Metrics
    memory_bandwidth: float = 0.0
    memory_read_bandwidth: float = 0.0
    memory_write_bandwidth: float = 0.0
    
    # Complete dataset analysis fields
    hardware_configuration: Optional[Dict[str, Any]] = None
    enum_mappings: Optional[Dict[str, Dict[int, str]]] = None
    performance_events: Optional[Dict[str, Any]] = None
    
    # Distribution analysis fields
    distribution_stats: Optional[Dict[str, Any]] = None
    per_rank_breakdown: Optional[Dict[str, Any]] = None
    per_thread_breakdown: Optional[Dict[str, Any]] = None
    per_stream_breakdown: Optional[Dict[str, Any]] = None
    outliers: Optional[Dict[str, Any]] = None
    bottleneck_details: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs):
        """Custom init to handle extra keyword arguments gracefully"""
        # Get list of valid fields for this dataclass
        import dataclasses
        names = set(f.name for f in dataclasses.fields(self))
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        self.__post_init__()
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        mpip_frac = float(self.mpip_max_mpi_wall_fraction or 0.0)
        if mpip_frac > 0.001:
            self.mpi_comm_fraction = min(1.0, max(0.0, mpip_frac))
        elif self.total_runtime > 0:
            self.mpi_comm_fraction = min(self.mpi_comm_time / self.total_runtime, 1.0)
        else:
            self.mpi_comm_fraction = 0.0

        if self.total_runtime > 0:
            self.thread_stall_fraction = min(self.thread_stall_time / self.total_runtime, 1.0)
        else:
            self.thread_stall_fraction = 0.0
            
        if self.memory_total_accesses > 0:
            self.locality_efficiency = self.memory_local_accesses / self.memory_total_accesses
        elif self.numa_efficiency > 0:
            # LIKWID provides numa_efficiency when local/remote counts aren't in the dict
            self.locality_efficiency = self.numa_efficiency
        else:
            self.locality_efficiency = 0.0
            
        # Thread stall as complement of CPU utilization
        self.thread_stall_fraction = max(1.0 - self.cpu_utilization, 0.0)
    
    def get_complete_mpi_analysis(self) -> Dict[str, Any]:
        """Get complete MPI analysis from performance events"""
        if not self.performance_events:
            return {}
        
        osrt_events = self.performance_events.get('osrt_api_events', {})
        if not osrt_events:
            return {}
        
        # Extract MPI events by eventClass (27 = OSRuntime/MPI)
        mpi_analysis = {}
        event_classes = osrt_events.get('event_classes', {})
        
        if 27 in event_classes:  # MPI events
            mpi_class = event_classes[27]
            mpi_analysis = {
                'mpi_event_count': mpi_class.get('event_count', 0),
                'mpi_total_duration': mpi_class.get('total_duration', 0.0),
                'mpi_events': mpi_class.get('events', [])
            }
        
        return mpi_analysis
    
    def get_complete_gpu_analysis(self) -> Dict[str, Any]:
        """Get complete GPU analysis from performance events"""
        if not self.performance_events:
            return {}
        
        gpu_analysis = {}
        
        # CUDA kernel analysis
        cuda_kernels = self.performance_events.get('cuda_kernel_events', {})
        if cuda_kernels:
            gpu_analysis['kernels'] = cuda_kernels
        
        # CUDA memory copy analysis
        cuda_memcpy = self.performance_events.get('cuda_memcpy_events', {})
        if cuda_memcpy:
            gpu_analysis['memcpy'] = cuda_memcpy
        
        return gpu_analysis
    
    def get_complete_thread_analysis(self) -> Dict[str, Any]:
        """Get complete thread analysis from performance events"""
        if not self.performance_events:
            return {}
        
        thread_analysis = {}
        
        # Scheduling events analysis
        sched_events = self.performance_events.get('sched_events', {})
        if sched_events:
            thread_analysis['scheduling'] = sched_events
        
        # Composite events analysis
        composite_events = self.performance_events.get('composite_events', {})
        if composite_events:
            thread_analysis['composite'] = composite_events
        
        return thread_analysis

@dataclass
class Configuration:
    """MPI+OpenMP configuration specification"""
    name: str
    mpi_ranks_per_node: int
    omp_threads_per_rank: int
    binding_strategy: str = "close"
    placement_strategy: str = "cores"
    
    def __str__(self):
        return f"{self.name} ({self.mpi_ranks_per_node}x{self.omp_threads_per_rank})"

class ConfigurationScorer:
    """
    Implements the heuristic scoring system for MPI+OpenMP configurations
    
    The scoring function evaluates configurations based on three key metrics:
    1. Communication Efficiency (MPI)
    2. Thread Efficiency (OpenMP)
    3. Memory Locality (NUMA) - requires LIKWID
    """
    
    def __init__(self, alpha: float = 0.30, beta: float = 0.30, gamma: float = 0.15, 
                 delta: float = 0.15, epsilon: float = 0.10, enable_locality: bool = False,
                 cpu_only: bool = False):
        """
        Initialize the scorer with customizable weights (5 components)
        
        Args:
            alpha: Weight for communication efficiency (1-MPI_comm)
            beta: Weight for thread efficiency (1-Thread_stall)
            gamma: Weight for memory locality (NUMA) - requires LIKWID
            delta: Weight for GPU utilization (GPU_util)
            epsilon: Weight for OpenMP efficiency (OpenMP_eff)
            enable_locality: Whether to use locality scoring (requires LIKWID data)
            cpu_only: If True, redistributes GPU and OpenMP weights to CPU metrics
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma if enable_locality else 0.0
        self.delta = delta if not cpu_only else 0.0
        self.epsilon = epsilon if not cpu_only else 0.0
        self.enable_locality = enable_locality
        self.cpu_only = cpu_only
        
        # Dynamic Baseline for Locality Scoring
        self.reference_bandwidth = 0.0
        
        
        # If locality disabled, redistribute gamma weight
        if not enable_locality and gamma > 0:
            # Redistribute gamma to alpha and beta equally
            redistribution = gamma / 2
            self.alpha += redistribution
            self.beta += redistribution
            logger.info(f"Locality disabled (no LIKWID data). Redistributing γ={gamma:.2f} to α and β")
        
        # If CPU-only mode, redistribute delta and epsilon weights
        if cpu_only and (delta > 0 or epsilon > 0):
            # Redistribute to alpha and beta proportionally to their current values
            total_to_redistribute = delta + epsilon
            alpha_portion = self.alpha / (self.alpha + self.beta)
            beta_portion = self.beta / (self.alpha + self.beta)
            self.alpha += total_to_redistribute * alpha_portion
            self.beta += total_to_redistribute * beta_portion
            logger.info(f"CPU-only mode. Redistributing δ={delta:.2f} and ε={epsilon:.2f} to α and β")
        
        # Validate weights sum to 1.0
        total_weight = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.alpha /= total_weight
            self.beta /= total_weight
            self.gamma /= total_weight
            self.delta /= total_weight
            self.epsilon /= total_weight
        
        if enable_locality:
            logger.info(f"Initialized scorer with weights: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}, δ={self.delta:.2f}, ε={self.epsilon:.2f}")
            logger.info(f"Locality scoring ENABLED (LIKWID data required)")
        else:
            logger.info(f"Initialized scorer with weights: α={self.alpha:.2f}, β={self.beta:.2f}, δ={self.delta:.2f}, ε={self.epsilon:.2f}")
            logger.info(f"Locality scoring DISABLED (use --enable-likwid to enable)")
        
        # Store scored configurations
        self.scored_configurations: Dict[str, Dict[str, Any]] = {}

    def set_reference_bandwidth(self, max_bandwidth: float):
        """
        Set the reference (maximum) memory bandwidth for dynamic scoring.
        
        Args:
            max_bandwidth: Maximum observed bandwidth in GB/s
        """
        self.reference_bandwidth = max_bandwidth
        logger.info(f"Set reference memory bandwidth for scoring: {max_bandwidth:.2f} GB/s")
        
    def score_configuration(self, 
                          config_name: str,
                          mpi_ranks: int,
                          omp_threads: int,
                          performance_data: Dict[str, Any]) -> float:
        """
        Score a configuration using the composite heuristic function
        
        Args:
            config_name: Name identifier for the configuration
            mpi_ranks: Number of MPI ranks per node
            omp_threads: Number of OpenMP threads per rank
            performance_data: Dictionary containing performance metrics
            
        Returns:
            Heuristic score (0.0 to 1.0, higher is better)
        """
        try:
            # Sanitize performance_data before constructing PerformanceMetrics.
            # If total_runtime=0 or mpi_comm_time=None the dataclass silently
            # sets mpi_comm_fraction=0.0, making comm_score meaningless for
            # configs that had no MPI overhead.  Clamp or default these fields.
            safe_data = dict(performance_data)
            total_rt = float(safe_data.get('total_runtime') or 0.0)
            mpi_ct   = float(safe_data.get('mpi_comm_time') or 0.0)
            numa_eff = float(safe_data.get('numa_efficiency') or 0.0)
            cpu_util = float(safe_data.get('cpu_utilization') or 0.0)

            # If runtime is missing/zero use wall-clock runtime if available
            if total_rt <= 0:
                total_rt = float(safe_data.get('runtime_sec') or
                                 safe_data.get('wall_clock_time') or 0.0)

            safe_data['total_runtime']   = total_rt
            safe_data['mpi_comm_time']   = min(mpi_ct, total_rt) if total_rt > 0 else mpi_ct
            safe_data['numa_efficiency'] = numa_eff
            safe_data['cpu_utilization'] = cpu_util

            logger.debug(f"[score_configuration] {config_name}: "
                         f"runtime={total_rt:.2f}s, mpi_comm={safe_data['mpi_comm_time']:.3f}s, "
                         f"numa={numa_eff:.3f}, cpu={cpu_util:.3f}")

            # Convert performance data to metrics object
            metrics = PerformanceMetrics(**safe_data)
            
            # Calculate individual component scores with configuration awareness
            comm_score = self._calculate_communication_score(metrics, mpi_ranks, omp_threads)
            thread_score = self._calculate_thread_score(metrics, mpi_ranks, omp_threads)
            gpu_score = self._calculate_gpu_score(metrics, mpi_ranks, omp_threads)
            openmp_score = self._calculate_openmp_score(metrics, mpi_ranks, omp_threads)
            
            # Calculate locality score only if enabled (requires LIKWID data)
            locality_score = 0.0
            if self.enable_locality:
                locality_score = self._calculate_locality_score(metrics, mpi_ranks, omp_threads)
            
            # Adaptive weight redistribution for pure configurations
            # For pure OpenMP (1xN): MPI communication is irrelevant, redistribute alpha to epsilon
            # For pure MPI (Nx1): OpenMP efficiency is irrelevant, redistribute epsilon to alpha
            is_pure_openmp = (mpi_ranks == 1 and omp_threads > 1)
            is_pure_mpi = (mpi_ranks > 1 and omp_threads == 1)
            
            if is_pure_openmp:
                # Pure OpenMP: Redistribute alpha weight to epsilon (MPI is irrelevant)
                effective_alpha = 0.0
                effective_epsilon = self.epsilon + self.alpha
                effective_beta = self.beta
                effective_gamma = self.gamma
                effective_delta = self.delta
            elif is_pure_mpi:
                # Pure MPI: Redistribute epsilon weight to alpha (OpenMP is irrelevant)
                effective_alpha = self.alpha + self.epsilon
                effective_epsilon = 0.0
                effective_beta = self.beta
                effective_gamma = self.gamma
                effective_delta = self.delta
            else:
                # Hybrid or other: Use original weights
                effective_alpha = self.alpha
                effective_epsilon = self.epsilon
                effective_beta = self.beta
                effective_gamma = self.gamma
                effective_delta = self.delta
            
            # Normalize weights to ensure they sum to 1.0
            total_weight = effective_alpha + effective_beta + effective_gamma + effective_delta + effective_epsilon
            if total_weight > 0:
                effective_alpha /= total_weight
                effective_beta /= total_weight
                effective_gamma /= total_weight
                effective_delta /= total_weight
                effective_epsilon /= total_weight
            
            # Apply adaptive weights and calculate composite score
            composite_score = (effective_alpha * comm_score + 
                             effective_beta * thread_score + 
                             effective_gamma * locality_score +
                             effective_delta * gpu_score +
                             effective_epsilon * openmp_score)
            
            # Store scoring details
            self.scored_configurations[config_name] = {
                'mpi_ranks': mpi_ranks,
                'omp_threads': omp_threads,
                'composite_score': composite_score,
                'component_scores': {
                    'communication': comm_score,
                    'threading': thread_score,
                    'locality': locality_score,
                    'gpu': gpu_score,
                    'openmp': openmp_score
                },
                'metrics': performance_data,
                'detailed_breakdown': {
                    'mpi_comm_fraction': metrics.mpi_comm_fraction,
                    'thread_stall_fraction': metrics.thread_stall_fraction,
                    'locality_efficiency': metrics.locality_efficiency if self.enable_locality else 0.0,
                    'cpu_utilization': metrics.cpu_utilization
                }
            }
            
            if self.enable_locality:
                logger.info(f"Scored {config_name}: {composite_score:.4f} "
                           f"(Comm: {comm_score:.3f}, Thread: {thread_score:.3f}, Locality: {locality_score:.3f})")
            else:
                logger.info(f"Scored {config_name}: {composite_score:.4f} "
                           f"(Comm: {comm_score:.3f}, Thread: {thread_score:.3f})")
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Error scoring configuration {config_name}: {e}")
            return 0.0
    
    def _calculate_communication_score(self, metrics: PerformanceMetrics, mpi_ranks: int, omp_threads: int) -> float:
        """
        Calculate communication efficiency score with configuration awareness
        
        Args:
            metrics: Performance metrics
            mpi_ranks: Number of MPI ranks
            omp_threads: Number of OpenMP threads per rank
            
        Returns:
            Communication score (0.0 to 1.0)
        """
        # Base communication score from MPI communication fraction
        base_comm_score = 1.0 - metrics.mpi_comm_fraction
        
        # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
        is_pure_mpi = (mpi_ranks > 1 and omp_threads == 1)
        is_pure_openmp = (mpi_ranks == 1 and omp_threads > 1)
        is_hybrid = (mpi_ranks > 1 and omp_threads > 1)
        
        # Configuration-based adjustments
        if is_pure_openmp:
            # Pure OpenMP: No MPI communication, so communication efficiency is perfect
            comm_score = 1.0
        elif is_pure_mpi:
            # Pure MPI: Communication overhead scales with number of ranks
            # More ranks = more communication, but this is expected and accounted for in mpi_comm_fraction
            # Apply a small penalty for very high rank counts (communication complexity)
            if mpi_ranks > 16:
                # For very high rank counts, slight penalty for communication complexity
                rank_complexity_penalty = min(0.15, (mpi_ranks - 16) * 0.01)  # Max 15% penalty
                comm_score = base_comm_score * (1.0 - rank_complexity_penalty)
            else:
                comm_score = base_comm_score
        elif is_hybrid:
            # Hybrid: MPI communication overhead + potential for communication/computation overlap
            # More ranks = more communication overhead
            mpi_overhead_penalty = min(0.20, (mpi_ranks - 1) * 0.02)  # Max 20% penalty, 2% per additional rank
            
            # In hybrid, more threads can help with communication/computation overlap
            # Use a gradual bonus based on thread count (no arbitrary threshold)
            # More threads = better potential for overlap, but diminishing returns
            if omp_threads > 1:
                # Gradual bonus: sqrt(threads/2) gives diminishing returns, capped at 10%
                overlap_bonus = min(0.10, np.sqrt(omp_threads / 2.0) * 0.05)
            else:
                overlap_bonus = 0.0
            
            comm_score = base_comm_score * (1.0 - mpi_overhead_penalty + overlap_bonus)
        else:
            # Fallback: single rank, single thread (shouldn't happen in practice)
            comm_score = base_comm_score
        
        # Ensure score is in valid range
        comm_score = max(0.0, min(1.0, comm_score))
        
        return comm_score
    
    def _calculate_thread_score(self, metrics: PerformanceMetrics, mpi_ranks: int, omp_threads: int) -> float:
        """
        Calculate thread efficiency score with configuration awareness
        
        Args:
            metrics: Performance metrics
            mpi_ranks: Number of MPI ranks
            omp_threads: Number of OpenMP threads per rank
            
        Returns:
            Thread score (0.0 to 1.0)
        """
        # Base thread score from CPU utilization (Nsight). When stdout-only metrics leave cpu=0,
        # do not treat as "0% utilized" — use neutral 0.5 until a trace exists.
        base_thread_score = metrics.cpu_utilization
        if (not getattr(metrics, "profiling_trace_present", True)) and base_thread_score <= 0.01:
            base_thread_score = 0.5
        
        # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
        is_pure_mpi = (mpi_ranks > 1 and omp_threads == 1)
        is_pure_openmp = (mpi_ranks == 1 and omp_threads > 1)
        is_hybrid = (mpi_ranks > 1 and omp_threads > 1)
        
        # Fix 1: Thread balance factor - only applies when OpenMP parallelism exists
        if is_pure_mpi:
            # Pure MPI: No OpenMP threads to balance, so no penalty
            thread_balance_factor = 1.0
        elif is_pure_openmp or is_hybrid:
            # OpenMP parallelism exists: Use gradual scaling instead of arbitrary 4.0 threshold
            # More threads = better load balancing, but with diminishing returns
            if omp_threads >= 4:
                # 4+ threads: Full credit (1.0)
                thread_balance_factor = 1.0
            else:
                # 1-3 threads: Gradual scaling (not harsh penalty)
                thread_balance_factor = 0.7 + (omp_threads - 1) * 0.1  # 0.7, 0.8, 0.9 for 1,2,3 threads
            
            # Fix 4: Upper bound penalty for excessive threads (context switching, cache contention)
            if omp_threads > 32:
                # Penalty for very high thread counts
                excess_penalty = min(0.15, (omp_threads - 32) * 0.005)  # Max 15% penalty
                thread_balance_factor *= (1.0 - excess_penalty)
        else:
            # Fallback: single rank, single thread
            thread_balance_factor = 1.0
        
        # Fix 2: MPI sync overhead - only applies when MPI parallelism exists
        if is_pure_openmp:
            # Pure OpenMP: No MPI synchronization, so no penalty
            sync_overhead_factor = 1.0
        elif is_pure_mpi or is_hybrid:
            # MPI parallelism exists: Sync overhead scales with ranks
            # Use logarithmic scaling instead of linear to avoid floor issues
            if mpi_ranks == 1:
                sync_overhead_factor = 1.0
            else:
                # Logarithmic penalty: log2(ranks) gives better scaling than linear
                # 2 ranks: ~0.99, 4 ranks: ~0.98, 8 ranks: ~0.97, 16 ranks: ~0.95, 48 ranks: ~0.92
                log_penalty = np.log2(mpi_ranks) * 0.01  # 1% per doubling of ranks
                sync_overhead_factor = max(0.7, 1.0 - log_penalty)  # Lower floor (0.7) for very high ranks
        else:
            sync_overhead_factor = 1.0
        
        # Fix 5: Use weighted combination instead of multiplicative (less harsh)
        # This prevents one bad factor from dominating the entire score
        thread_score = base_thread_score * (0.7 * thread_balance_factor + 0.3 * sync_overhead_factor)
        
        return max(0.0, min(1.0, thread_score))
    
    def _calculate_locality_score(self, metrics: PerformanceMetrics, mpi_ranks: int, omp_threads: int) -> float:
        """
        Calculate memory locality score with configuration awareness
        
        Separates cache locality (OpenMP-related) from NUMA locality (MPI-related)
        
        Args:
            metrics: Performance metrics
            mpi_ranks: Number of MPI ranks
            omp_threads: Number of OpenMP threads per rank
            
        Returns:
            Locality score (0.0 to 1.0)
        """
        # Base locality score from NUMA efficiency (from LIKWID data)
        base_locality_score = metrics.locality_efficiency
        
        # FIX: Use Dynamic Bandwidth Scoring if available (more robust than NUMA efficiency)
        if hasattr(self, 'reference_bandwidth') and self.reference_bandwidth > 0 and metrics.memory_bandwidth > 0:
            # Score = Raw Bandwidth / Reference Bandwidth
            # This self-calibrates to the hardware capabilities
            bandwidth_score = metrics.memory_bandwidth / self.reference_bandwidth
            # Cap at 1.0 (in case specific run exceeds reference slightly)
            # Use small tolerance
            bandwidth_score = min(1.0, bandwidth_score / 0.95)
            bandwidth_score = min(1.0, bandwidth_score)
            
            # Use Bandwidth Score as the base, fallback to NUMA if bandwidth is missing
            base_locality_score = bandwidth_score
            logger.debug(f"  {mpi_ranks}x{omp_threads}: Using Bandwidth Score {bandwidth_score:.3f} (BW={metrics.memory_bandwidth:.2f} GB/s / Ref={self.reference_bandwidth:.2f} GB/s)")
        
        # If base score is 0 (no LIKWID data), return 0
        
        # If base score is 0 (no LIKWID data), return 0
        if base_locality_score <= 0.0:
            return 0.0
        
        # Differentiate between pure MPI, pure OpenMP, and hybrid scenarios
        is_pure_mpi = (mpi_ranks > 1 and omp_threads == 1)
        is_pure_openmp = (mpi_ranks == 1 and omp_threads > 1)
        is_hybrid = (mpi_ranks > 1 and omp_threads > 1)
        
        # Fix 1 & 7: Separate cache locality (OpenMP) from NUMA locality (MPI)
        # Cache locality: How well threads share cache (OpenMP-related)
        if is_pure_mpi:
            # Pure MPI: Cache locality not applicable (process-level, not thread-level)
            cache_locality_factor = 1.0  # No penalty
        elif is_pure_openmp or is_hybrid:
            # OpenMP parallelism exists: Cache locality matters
            # Use gradual scaling instead of arbitrary 8.0 threshold
            if omp_threads <= 8:
                # 1-8 threads: Gradual scaling
                cache_locality_factor = 0.7 + (omp_threads - 1) * (0.3 / 7.0)  # 0.7 to 1.0
            elif omp_threads <= 32:
                # 9-32 threads: Optimal range (full credit)
                cache_locality_factor = 1.0
            else:
                # Fix 4: Upper bound penalty for excessive threads (cache thrashing, false sharing)
                # >32 threads: Penalty for too many threads
                excess_penalty = min(0.20, (omp_threads - 32) * 0.005)  # Max 20% penalty
                cache_locality_factor = 1.0 - excess_penalty
        else:
            # Fallback: single rank, single thread
            cache_locality_factor = 1.0
        
        # Fix 2: NUMA locality (MPI-related) - only applies when MPI parallelism exists
        # NUMA locality: How well processes use local memory (MPI-related)
        if is_pure_openmp:
            # Pure OpenMP: No MPI data distribution, so NUMA factor doesn't apply
            memory_distribution_factor = 1.0  # No penalty
        elif is_pure_mpi or is_hybrid:
            # MPI parallelism exists: NUMA distribution matters
            # Fix 3: Use logarithmic scaling instead of linear to avoid floor issues
            if mpi_ranks == 1:
                memory_distribution_factor = 1.0
            else:
                # Logarithmic penalty: log2(ranks) gives better scaling than linear
                # 2 ranks: ~0.97, 4 ranks: ~0.94, 8 ranks: ~0.91, 16 ranks: ~0.85, 48 ranks: ~0.76
                log_penalty = np.log2(mpi_ranks) * 0.03  # 3% per doubling of ranks
                memory_distribution_factor = max(0.7, 1.0 - log_penalty)  # Lower floor (0.7) for very high ranks
        else:
            memory_distribution_factor = 1.0
        
        # Fix 5: Apply only relevant factors based on configuration type (not multiplicative)
        if is_pure_openmp:
            # Pure OpenMP: Only cache locality matters
            locality_score = base_locality_score * cache_locality_factor
        elif is_pure_mpi:
            # Pure MPI: Only NUMA distribution matters
            locality_score = base_locality_score * memory_distribution_factor
        else:
            # Hybrid: Both matter, but use weighted combination (not multiplication)
            # This prevents one bad factor from dominating
            locality_score = base_locality_score * (0.6 * cache_locality_factor + 0.4 * memory_distribution_factor)
        
        return max(0.0, min(1.0, locality_score))
    
    def _calculate_gpu_score(self, metrics: PerformanceMetrics, mpi_ranks: int, omp_threads: int) -> float:
        """
        Calculate GPU utilization score with configuration awareness
        
        Args:
            metrics: Performance metrics
            mpi_ranks: Number of MPI ranks
            omp_threads: Number of OpenMP threads per rank
            
        Returns:
            GPU score (0.0 to 1.0)
        """
        # Fix 8: Handle missing or invalid GPU data
        gpu_utilization = metrics.gpu_utilization
        
        # Check if GPU data is available (None, negative, or NaN indicates missing data)
        if gpu_utilization is None or gpu_utilization < 0 or np.isnan(gpu_utilization):
            # Missing data - return 0.0 (treat as no GPU utilization)
            return 0.0
        
        # Fix 2: Handle CPU-only configs explicitly
        # If utilization is exactly 0.0 and we're in CPU-only mode, return 0.0
        if gpu_utilization == 0.0:
            return 0.0
        
        # Base score from GPU compute utilization
        base_gpu_score = gpu_utilization
        
        # Fix 5: Consider GPU memory bandwidth if available
        if hasattr(metrics, 'gpu_memory_bandwidth') and metrics.gpu_memory_bandwidth is not None:
            if metrics.gpu_memory_bandwidth > 0:
                # Normalize memory bandwidth (assuming max ~1000 GB/s for high-end GPUs)
                # Adjust this threshold based on your GPU hardware
                memory_util = min(1.0, metrics.gpu_memory_bandwidth / 1000.0)
                # Combine compute and memory utilization (weighted average)
                base_gpu_score = 0.7 * base_gpu_score + 0.3 * memory_util
        
        # Fix 1 & 4: Configuration-based adjustments for multi-GPU setups
        # For multi-rank configurations, GPU utilization might be per-rank
        # In multi-GPU setups, more ranks can utilize more GPUs
        if mpi_ranks > 1:
            # Multi-rank: Could indicate multi-GPU setup
            # If utilization is already aggregated across ranks, use as-is
            # If per-rank, we'd need to aggregate (assume aggregated for now)
            # Small bonus for multi-GPU efficiency (if utilization is already high)
            if base_gpu_score > 0.7:
                # High utilization with multiple ranks suggests good multi-GPU scaling
                multi_gpu_bonus = min(0.05, (mpi_ranks - 1) * 0.002)  # Max 5% bonus
                base_gpu_score = min(1.0, base_gpu_score + multi_gpu_bonus)
        
        # Fix 6: Consider CPU-GPU coordination for hybrid configs
        if omp_threads > 1 and hasattr(metrics, 'cpu_utilization'):
            cpu_util = metrics.cpu_utilization
            # If GPU is high but CPU is very low, might indicate imbalance
            if base_gpu_score > 0.8 and cpu_util < 0.3:
                # Slight penalty for CPU-GPU imbalance
                imbalance_penalty = 0.1
                base_gpu_score *= (1.0 - imbalance_penalty)
            # If both GPU and CPU are high, slight bonus for good coordination
            elif base_gpu_score > 0.7 and cpu_util > 0.7:
                coordination_bonus = 0.05
                base_gpu_score = min(1.0, base_gpu_score + coordination_bonus)
        
        # Fix 3 & 7: Remove sigmoid transformation (use linear for interpretability)
        # Linear mapping: 0.0 -> 0.0, 1.0 -> 1.0 (exact, interpretable)
        gpu_score = base_gpu_score
        
        return max(0.0, min(1.0, gpu_score))
    
    def _calculate_openmp_score(self, metrics: PerformanceMetrics, mpi_ranks: int, omp_threads: int) -> float:
        """
        Calculate OpenMP work efficiency score with configuration awareness
        
        Args:
            metrics: Performance metrics
            mpi_ranks: Number of MPI ranks
            omp_threads: Number of OpenMP threads per rank
            
        Returns:
            OpenMP score (0.0 to 1.0)
        """
        # Fix 3: Handle pure MPI case - epsilon shouldn't apply
        if omp_threads == 1:
            # Pure MPI: No OpenMP parallelism, efficiency is N/A
            # Return 1.0 (perfect score) since there's no OpenMP overhead
            return 1.0
        
        # Fix 6: Handle missing or invalid OpenMP data
        base_openmp_score = metrics.openmp_work_efficiency
        
        trace_missing = not getattr(metrics, "profiling_trace_present", True)
        # stdout-only path uses 0.0 for unknown trace-derived OpenMP efficiency — treat as missing
        eff_invalid = (
            base_openmp_score is None
            or base_openmp_score < 0
            or np.isnan(base_openmp_score)
            or (trace_missing and base_openmp_score <= 0.01 and omp_threads > 1)
        )
        if eff_invalid:
            # Missing trace-derived ε — use CPU utilization when available
            cu = getattr(metrics, "cpu_utilization", None)
            if cu is not None and cu > 0.01:
                base_openmp_score = float(cu)
            else:
                # No trace and no CPU signal — neutral score for ε
                return 0.5
        
        # Fix 7 & 8: Configuration-based adjustments for thread count
        # More threads can improve work-stealing (up to a point)
        if omp_threads <= 8:
            # 1-8 threads: Gradual scaling (not harsh penalty)
            thread_factor = 0.7 + (omp_threads - 1) * (0.3 / 7.0)  # 0.7 to 1.0
        elif omp_threads <= 32:
            # 9-32 threads: Optimal range (full credit)
            thread_factor = 1.0
        else:
            # Fix 8: Upper bound penalty for excessive threads (overhead, contention)
            # >32 threads: Penalty for too many threads
            excess_penalty = min(0.20, (omp_threads - 32) * 0.005)  # Max 20% penalty
            thread_factor = 1.0 - excess_penalty
        
        # Fix 2: Consider MPI ranks for work distribution
        # More ranks can affect work distribution and coordination
        if mpi_ranks > 1:
            # Hybrid: More ranks = more work distribution opportunities
            # But also more coordination overhead
            # Use logarithmic scaling to avoid harsh penalties
            if mpi_ranks <= 8:
                rank_factor = 1.0 - (mpi_ranks - 1) * 0.01  # Small penalty
            else:
                log_penalty = np.log2(mpi_ranks) * 0.01  # Logarithmic penalty
                rank_factor = 1.0 - log_penalty
            rank_factor = max(0.85, rank_factor)  # Minimum 85% score
        else:
            # Pure OpenMP: No MPI coordination overhead
            rank_factor = 1.0
        
        # Fix 5: Use weighted combination instead of sigmoid (more interpretable)
        # Combine base efficiency with configuration factors
        openmp_score = base_openmp_score * (0.7 * thread_factor + 0.3 * rank_factor)
        
        return max(0.0, min(1.0, openmp_score))
    
    def get_best_configuration(self) -> Optional[Tuple[str, float]]:
        """
        Get the configuration with the highest score
        
        Returns:
            Tuple of (config_name, score) or None if no configurations scored
        """
        if not self.scored_configurations:
            return None
            
        best_config = max(self.scored_configurations.items(), 
                         key=lambda x: x[1]['composite_score'])
        
        return best_config[0], best_config[1]['composite_score']
    
    def get_configuration_ranking(self) -> List[Tuple[str, float, Dict]]:
        """
        Get all configurations ranked by score (highest first)
        
        Returns:
            List of (config_name, score, details) tuples
        """
        if not self.scored_configurations:
            return []
            
        ranked = [(name, data['composite_score'], data) 
                  for name, data in self.scored_configurations.items()]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def get_detailed_analysis(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed analysis for a specific configuration
        
        Args:
            config_name: Name of the configuration to analyze
            
        Returns:
            Detailed analysis dictionary or None if not found
        """
        if config_name not in self.scored_configurations:
            return None
            
        config_data = self.scored_configurations[config_name]
        
        # Calculate performance improvement over worst configuration
        scores = [data['composite_score'] for data in self.scored_configurations.values()]
        worst_score = min(scores)
        best_score = max(scores)
        
        if worst_score > 0:
            improvement_percent = ((config_data['composite_score'] - worst_score) / worst_score) * 100
        else:
            improvement_percent = 0.0
            
        # Calculate rank among all configurations
        ranked_configs = self.get_configuration_ranking()
        rank = next((i for i, (name, _, _) in enumerate(ranked_configs) if name == config_name), -1)
        
        analysis = {
            'configuration': config_name,
            'mpi_ranks': config_data['mpi_ranks'],
            'omp_threads': config_data['omp_threads'],
            'composite_score': config_data['composite_score'],
            'rank': rank + 1,  # 1-based ranking
            'total_configurations': len(self.scored_configurations),
            'performance_improvement': improvement_percent,
            'component_scores': config_data['component_scores'],
            'detailed_breakdown': config_data['detailed_breakdown'],
            'recommendations': self._generate_recommendations(config_data)
        }
        
        return analysis
    
    def _generate_recommendations(self, config_data: Dict[str, Any]) -> List[str]:
        """
        Generate optimization recommendations based on component scores and distribution analysis
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        metrics = config_data.get('metrics', {})
        
        # Get bottleneck details from distribution analysis
        bottleneck_details = metrics.get('bottleneck_details', {})
        per_rank_breakdown = metrics.get('per_rank_breakdown', {})
        per_thread_breakdown = metrics.get('per_thread_breakdown', {})
        per_stream_breakdown = metrics.get('per_stream_breakdown', {})
        
        # Communication recommendations with distribution analysis
        comm_score = config_data['component_scores']['communication']
        if comm_score < 0.6:
            recommendations.append("Consider reducing MPI communication frequency or message size")
            recommendations.append("Evaluate non-blocking MPI calls for communication overlap")
        
            # Add specific recommendations based on distribution analysis
            if bottleneck_details and bottleneck_details.get('communication_bottlenecks', {}).get('count', 0) > 0:
                comm_bottlenecks = bottleneck_details['communication_bottlenecks']
                affected_ranks = comm_bottlenecks.get('affected_ranks', [])
                if affected_ranks:
                    recommendations.append(
                        f"⚠️ CRITICAL: {len(affected_ranks)} MPI ranks show high communication overhead "
                        f"(ranks: {affected_ranks[:5]}{'...' if len(affected_ranks) > 5 else ''}). "
                        f"Redistribute work to balance communication load."
                    )
            
            # Check for high skewness in MPI distribution
            if per_rank_breakdown and per_rank_breakdown.get('distribution_stats', {}).get('skewness', 0) > 1.0:
                recommendations.append(
                    "High skewness detected in MPI communication distribution. "
                    "Consider load balancing or work redistribution across ranks."
                )
        
        # Threading recommendations with distribution analysis
        thread_score = config_data['component_scores']['threading']
        if thread_score < 0.7:
            recommendations.append("Review OpenMP scheduling strategy (dynamic vs static)")
            recommendations.append("Check for load imbalance between threads")
            recommendations.append("Consider adjusting chunk size for better load distribution")
            
            # Add specific recommendations based on distribution analysis
            if bottleneck_details and bottleneck_details.get('threading_bottlenecks', {}).get('count', 0) > 0:
                thread_bottlenecks = bottleneck_details['threading_bottlenecks']
                affected_threads = thread_bottlenecks.get('affected_threads', [])
                if affected_threads:
                    recommendations.append(
                        f"⚠️ CRITICAL: {len(affected_threads)} threads show low utilization "
                        f"(threads: {affected_threads[:5]}{'...' if len(affected_threads) > 5 else ''}). "
                        f"Optimize thread scheduling or work distribution."
                    )
            
            # Check for high variance in thread utilization
            if per_thread_breakdown and per_thread_breakdown.get('distribution_stats', {}).get('cv', 0) > 0.3:
                recommendations.append(
                    "High variance detected in thread utilization. "
                    "Consider dynamic scheduling or work-stealing for better load balance."
                )
        
        # Locality recommendations
        locality_score = config_data['component_scores']['locality']
        if locality_score < 0.6:
            recommendations.append("Review memory allocation strategy for NUMA awareness")
            recommendations.append("Consider using first-touch policy for memory placement")
            recommendations.append("Evaluate thread binding to NUMA domains")
        
        # GPU recommendations with distribution analysis
        gpu_score = config_data['component_scores'].get('gpu', 0.0)
        if gpu_score < 0.7:
            if bottleneck_details and bottleneck_details.get('gpu_bottlenecks', {}).get('count', 0) > 0:
                gpu_bottlenecks = bottleneck_details['gpu_bottlenecks']
                affected_streams = gpu_bottlenecks.get('affected_streams', [])
                if affected_streams:
                    recommendations.append(
                        f"⚠️ CRITICAL: {len(affected_streams)} GPU streams show high utilization "
                        f"(streams: {affected_streams[:5]}{'...' if len(affected_streams) > 5 else ''}). "
                        f"Balance GPU work across streams for better utilization."
                    )
            
            # Check for high skewness in GPU distribution
            if per_stream_breakdown and per_stream_breakdown.get('distribution_stats', {}).get('skewness', 0) > 1.0:
                recommendations.append(
                    "High skewness detected in GPU stream utilization. "
                    "Consider redistributing GPU kernels across streams."
                )
        
        # General recommendations
        if config_data['composite_score'] < 0.5:
            recommendations.append("Configuration shows significant optimization opportunities")
            recommendations.append("Consider alternative MPI/OpenMP ratios")
        
        # Overall bottleneck severity warning
        if bottleneck_details and bottleneck_details.get('overall_severity') == 'high':
            recommendations.insert(0, 
                "🚨 HIGH SEVERITY: Multiple bottlenecks detected. "
                "Review detailed bottleneck analysis in report."
            )
        
        return recommendations
    
    def export_results(self, output_path: Path) -> None:
        """
        Export scoring results to CSV and JSON files
        
        Args:
            output_path: Directory to save export files
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        csv_data = []
        for name, data in self.scored_configurations.items():
            row = {
                'configuration': name,
                'mpi_ranks': data['mpi_ranks'],
                'omp_threads': data['omp_threads'],
                'composite_score': data['composite_score'],
                'comm_score': data['component_scores']['communication'],
                'thread_score': data['component_scores']['threading'],
                'locality_score': data['component_scores']['locality'],
                'mpi_comm_fraction': data['detailed_breakdown']['mpi_comm_fraction'],
                'thread_stall_fraction': data['detailed_breakdown']['thread_stall_fraction'],
                'locality_efficiency': data['detailed_breakdown']['locality_efficiency'],
                'cpu_utilization': data['detailed_breakdown']['cpu_utilization']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = output_path / "configuration_scores.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Exported scores to {csv_file}")
        
        # Export to JSON
        json_file = output_path / "scoring_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.scored_configurations, f, indent=2, default=str)
        logger.info(f"Exported detailed results to {json_file}")
    
    def print_summary(self) -> None:
        """Print a summary of all scored configurations"""
        if not self.scored_configurations:
            print("No configurations have been scored yet.")
            return
            
        print("\n" + "="*80)
        print("MPI+OpenMP Configuration Scoring Summary")
        print("="*80)
        
        # Print header
        print(f"{'Configuration':<15} {'MPI':<4} {'OMP':<4} {'Score':<8} {'Comm':<6} {'Thread':<7} {'Locality':<8}")
        print("-" * 80)
        
        # Print ranked configurations
        ranked_configs = self.get_configuration_ranking()
        for i, (name, score, data) in enumerate(ranked_configs):
            comm_score = data['component_scores']['communication']
            thread_score = data['component_scores']['threading']
            locality_score = data['component_scores']['locality']
            
            print(f"{name:<15} {data['mpi_ranks']:<4} {data['omp_threads']:<4} "
                  f"{score:<8.4f} {comm_score:<6.3f} {thread_score:<7.3f} {locality_score:<8.3f}")
        
        print("-" * 80)
        
        # Print best configuration details
        best_name, best_score = self.get_best_configuration()
        if best_name:
            print(f"\n🏆 Best Configuration: {best_name} (Score: {best_score:.4f})")
            
            # Calculate improvement over worst
            worst_score = min(data['composite_score'] for data in self.scored_configurations.values())
            improvement = ((best_score - worst_score) / worst_score) * 100
            print(f"📈 Performance Improvement: {improvement:.1f}% over worst configuration")
            
            # Show detailed analysis
            analysis = self.get_detailed_analysis(best_name)
            if analysis:
                print(f"📊 Rank: {analysis['rank']}/{analysis['total_configurations']}")
                print(f"🔧 MPI Ranks: {analysis['mpi_ranks']}, OpenMP Threads: {analysis['omp_threads']}")

class AdaptiveConfigurationScorer(ConfigurationScorer):
    """
    Auto-adaptive heuristic scorer that adjusts weights based on performance analysis
    
    Uses rule-based adaptation (no ML/AI) based on:
    - Bottleneck detection
    - Statistical variance analysis
    - Performance correlation
    - Iterative refinement
    """
    
    def __init__(self, alpha: float = 0.30, beta: float = 0.30, gamma: float = 0.15,
                 delta: float = 0.15, epsilon: float = 0.10, 
                 enable_adaptation: bool = True, enable_locality: bool = False,
                 cpu_only: bool = False):
        """
        Initialize adaptive scorer (5 components, locality optional)
        
        Args:
            alpha: Initial weight for communication efficiency
            beta: Initial weight for thread efficiency
            gamma: Initial weight for memory locality (requires LIKWID)
            delta: Initial weight for GPU utilization
            epsilon: Initial weight for OpenMP efficiency
            enable_adaptation: Whether to enable adaptive weight adjustment
            enable_locality: Whether to use locality scoring (requires LIKWID data)
            cpu_only: If True, redistributes GPU and OpenMP weights to CPU metrics
        """
        super().__init__(alpha, beta, gamma, delta, epsilon, enable_locality, cpu_only)
        self.initial_weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'epsilon': self.epsilon
        }
        self.enable_adaptation = enable_adaptation
        self.adaptation_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized adaptive scorer (adaptation: {'enabled' if enable_adaptation else 'disabled'}, "
                   f"locality: {'enabled' if enable_locality else 'disabled'})")
    
    def adapt_weights_from_results(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Adapt weights based on performance results using rule-based analysis
        
        Args:
            all_results: Dictionary of all configuration results with metrics
            
        Returns:
            New adapted weights
        """
        if not self.enable_adaptation or not all_results:
            weights = {
                'alpha': self.alpha,
                'beta': self.beta,
                'delta': self.delta,
                'epsilon': self.epsilon
            }
            if self.enable_locality:
                weights['gamma'] = self.gamma
            return weights
        
        logger.info("🔄 Adapting heuristic weights based on performance analysis...")
        
        # Step 1: Identify primary bottleneck from all configurations
        bottleneck_analysis = self._analyze_bottlenecks(all_results)
        
        # Step 2: Calculate metric variance (high variance = important metric)
        metric_variance = self._calculate_metric_variance(all_results)
        
        # Step 3: Calculate performance correlation (which metrics correlate with performance)
        performance_correlation = self._calculate_performance_correlation(all_results)
        
        # Step 4: Rule-based weight adjustment
        new_weights = self._adjust_weights_rule_based(
            bottleneck_analysis, 
            metric_variance, 
            performance_correlation
        )
        
        # Step 5: Normalize weights
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        # Update weights
        self.alpha = new_weights['alpha']
        self.beta = new_weights['beta']
        if self.enable_locality:
            self.gamma = new_weights['gamma']
        self.delta = new_weights['delta']
        self.epsilon = new_weights['epsilon']
        
        self.adaptation_history.append({
            'iteration': len(self.adaptation_history) + 1,
            'weights': new_weights.copy(),
            'bottleneck': bottleneck_analysis['primary'],
            'reason': bottleneck_analysis['reason'],
            'metric_variance': metric_variance,
            'performance_correlation': performance_correlation
        })
        
        if self.enable_locality:
            logger.info(f"✅ Adapted weights: α={self.alpha:.3f}, β={self.beta:.3f}, "
                       f"γ={self.gamma:.3f}, δ={self.delta:.3f}, ε={self.epsilon:.3f}")
        else:
            logger.info(f"✅ Adapted weights: α={self.alpha:.3f}, β={self.beta:.3f}, "
                       f"δ={self.delta:.3f}, ε={self.epsilon:.3f}")
            
        logger.info(f"   Reason: {bottleneck_analysis['reason']}")
        
        return new_weights
    
    def _analyze_bottlenecks(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify primary bottleneck across all configurations"""
        bottleneck_counts = {
            'communication': 0,
            'threading': 0,
            'memory': 0,
            'gpu': 0,
            'openmp': 0
        }
        
        bottleneck_severities = {
            'communication': [],
            'threading': [],
            'memory': [],
            'gpu': [],
            'openmp': []
        }
        
        for config_name, result in all_results.items():
            metrics = result.get('metrics', {})
            bottleneck_details = metrics.get('bottleneck_details', {})
            
            # Count bottlenecks
            comm_bottlenecks = bottleneck_details.get('communication_bottlenecks', {})
            if comm_bottlenecks.get('count', 0) > 0:
                bottleneck_counts['communication'] += 1
                severity = comm_bottlenecks.get('severity', 0.5)
                bottleneck_severities['communication'].append(severity)
            
            thread_bottlenecks = bottleneck_details.get('threading_bottlenecks', {})
            if thread_bottlenecks.get('count', 0) > 0:
                bottleneck_counts['threading'] += 1
                severity = thread_bottlenecks.get('severity', 0.5)
                bottleneck_severities['threading'].append(severity)
            
            memory_bottlenecks = bottleneck_details.get('memory_bottlenecks', {})
            if memory_bottlenecks.get('count', 0) > 0:
                bottleneck_counts['memory'] += 1
                severity = memory_bottlenecks.get('severity', 0.5)
                bottleneck_severities['memory'].append(severity)
            
            gpu_bottlenecks = bottleneck_details.get('gpu_bottlenecks', {})
            if gpu_bottlenecks.get('count', 0) > 0:
                bottleneck_counts['gpu'] += 1
                severity = gpu_bottlenecks.get('severity', 0.5)
                bottleneck_severities['gpu'].append(severity)
        
        # Find primary bottleneck (most frequent)
        if not any(bottleneck_counts.values()):
            return {
                'primary': 'none',
                'count': 0,
                'severities': {},
                'reason': 'No significant bottlenecks detected'
            }
        
        primary_bottleneck = max(bottleneck_counts.items(), key=lambda x: x[1])
        
        # Calculate average severity
        avg_severity = {}
        for bottleneck, severities in bottleneck_severities.items():
            if severities:
                avg_severity[bottleneck] = np.mean(severities)
            else:
                avg_severity[bottleneck] = 0.0
        
        return {
            'primary': primary_bottleneck[0],
            'count': primary_bottleneck[1],
            'severities': avg_severity,
            'reason': f"Primary bottleneck: {primary_bottleneck[0]} (appears in {primary_bottleneck[1]} configs)"
        }
    
    def _calculate_metric_variance(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate variance of each metric (high variance = important for discrimination)"""
        metrics_data = {
            'mpi_comm': [],
            'thread_stall': [],
            'locality': [],
            'gpu_util': [],
            'openmp_eff': []
        }
        
        for config_name, result in all_results.items():
            metrics = result.get('metrics', {})
            total_runtime = max(metrics.get('total_runtime', 1.0), 0.001)
            
            mpi_comm_time = metrics.get('mpi_comm_time', 0.0)
            metrics_data['mpi_comm'].append(mpi_comm_time / total_runtime)
            
            cpu_util = metrics.get('cpu_utilization', 0.0)
            metrics_data['thread_stall'].append(1.0 - cpu_util)
            
            metrics_data['locality'].append(metrics.get('numa_efficiency', 0.0))
            metrics_data['gpu_util'].append(metrics.get('gpu_utilization', 0.0))
            metrics_data['openmp_eff'].append(metrics.get('openmp_work_efficiency', 0.0))
        
        variance = {}
        for metric, values in metrics_data.items():
            if len(values) > 1:
                variance[metric] = float(np.var(values))
            else:
                variance[metric] = 0.0
        
        return variance
    
    def _calculate_performance_correlation(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlation between metrics and actual performance (runtime)"""
        metrics_data = {
            'mpi_comm': [],
            'thread_stall': [],
            'locality': [],
            'gpu_util': [],
            'openmp_eff': []
        }
        runtimes = []
        
        for config_name, result in all_results.items():
            metrics = result.get('metrics', {})
            runtime = metrics.get('total_runtime', 0)
            if runtime > 0:
                runtimes.append(runtime)
                total_runtime = max(runtime, 0.001)
                
                mpi_comm_time = metrics.get('mpi_comm_time', 0.0)
                metrics_data['mpi_comm'].append(mpi_comm_time / total_runtime)
                
                cpu_util = metrics.get('cpu_utilization', 0.0)
                metrics_data['thread_stall'].append(1.0 - cpu_util)
                
                metrics_data['locality'].append(metrics.get('numa_efficiency', 0.0))
                metrics_data['gpu_util'].append(metrics.get('gpu_utilization', 0.0))
                metrics_data['openmp_eff'].append(metrics.get('openmp_work_efficiency', 0.0))
        
        correlations = {}
        for metric, values in metrics_data.items():
            if len(values) > 1 and len(runtimes) > 1 and len(values) == len(runtimes):
                try:
                    # For mpi_comm and thread_stall, negative correlation is good (lower = better performance)
                    # For locality, gpu_util, openmp_eff, positive correlation is good (higher = better performance)
                    if metric in ['mpi_comm', 'thread_stall']:
                        corr = -np.corrcoef(values, runtimes)[0, 1]  # Negative correlation
                    else:
                        corr = np.corrcoef(values, runtimes)[0, 1]  # Positive correlation
                    correlations[metric] = max(0, corr)  # Only positive correlations matter
                except:
                    correlations[metric] = 0.0
            else:
                correlations[metric] = 0.0
        
        return correlations
    
    def _adjust_weights_rule_based(self, bottleneck_analysis: Dict[str, Any], 
                                   metric_variance: Dict[str, float], 
                                   performance_correlation: Dict[str, float]) -> Dict[str, float]:
        """Rule-based weight adjustment"""
        # Start with current weights
        new_weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'delta': self.delta,
            'epsilon': self.epsilon
        }
        if self.enable_locality:
            new_weights['gamma'] = self.gamma
        
        # Rule 1: Increase weight for primary bottleneck
        primary_bottleneck = bottleneck_analysis.get('primary', 'none')
        if primary_bottleneck == 'communication':
            new_weights['alpha'] *= 1.5  # Increase MPI weight
            new_weights['beta'] *= 0.9
            if self.enable_locality:
                new_weights['gamma'] *= 0.9
            logger.debug("Rule 1: Increased α (communication) due to primary bottleneck")
        elif primary_bottleneck == 'threading':
            new_weights['beta'] *= 1.5  # Increase thread weight
            new_weights['alpha'] *= 0.9
            if self.enable_locality:
                new_weights['gamma'] *= 0.9
            logger.debug("Rule 1: Increased β (threading) due to primary bottleneck")
        elif primary_bottleneck == 'memory' and self.enable_locality:
            new_weights['gamma'] *= 1.5  # Increase memory weight
            new_weights['alpha'] *= 0.9
            new_weights['beta'] *= 0.9
            logger.debug("Rule 1: Increased γ (memory) due to primary bottleneck")
        elif primary_bottleneck == 'gpu':
            new_weights['delta'] *= 1.5  # Increase GPU weight
            new_weights['alpha'] *= 0.9
            new_weights['beta'] *= 0.9
            if self.enable_locality:
                new_weights['gamma'] *= 0.9
            logger.debug("Rule 1: Increased δ (GPU) due to primary bottleneck")
        
        # Rule 2: Increase weight for metrics with high variance (discriminating power)
        max_variance = max(metric_variance.values()) if metric_variance.values() else 1.0
        if max_variance > 0:
            for metric, variance in metric_variance.items():
                if variance > 0.5 * max_variance:  # High variance
                    if metric == 'mpi_comm':
                        new_weights['alpha'] *= 1.2
                        logger.debug(f"Rule 2: Increased α due to high variance in {metric}")
                    elif metric == 'thread_stall':
                        new_weights['beta'] *= 1.2
                        logger.debug(f"Rule 2: Increased β due to high variance in {metric}")
                    elif metric == 'locality' and self.enable_locality:
                        new_weights['gamma'] *= 1.2
                        logger.debug(f"Rule 2: Increased γ due to high variance in {metric}")
                    elif metric == 'gpu_util':
                        new_weights['delta'] *= 1.2
                        logger.debug(f"Rule 2: Increased δ due to high variance in {metric}")
                    elif metric == 'openmp_eff':
                        new_weights['epsilon'] *= 1.2
                        logger.debug(f"Rule 2: Increased ε due to high variance in {metric}")
        
        # Rule 3: Increase weight for metrics with high performance correlation
        max_correlation = max(performance_correlation.values()) if performance_correlation.values() else 1.0
        if max_correlation > 0:
            for metric, correlation in performance_correlation.items():
                if correlation > 0.5 * max_correlation and correlation > 0.3:  # High correlation
                    if metric == 'mpi_comm':
                        new_weights['alpha'] *= 1.3
                        logger.debug(f"Rule 3: Increased α due to high correlation in {metric} ({correlation:.3f})")
                    elif metric == 'thread_stall':
                        new_weights['beta'] *= 1.3
                        logger.debug(f"Rule 3: Increased β due to high correlation in {metric} ({correlation:.3f})")
                    elif metric == 'locality' and self.enable_locality:
                        new_weights['gamma'] *= 1.3
                        logger.debug(f"Rule 3: Increased γ due to high correlation in {metric} ({correlation:.3f})")
                    elif metric == 'gpu_util':
                        new_weights['delta'] *= 1.3
                        logger.debug(f"Rule 3: Increased δ due to high correlation in {metric} ({correlation:.3f})")
                    elif metric == 'openmp_eff':
                        new_weights['epsilon'] *= 1.3
                        logger.debug(f"Rule 3: Increased ε due to high correlation in {metric} ({correlation:.3f})")
        
        # Rule 4: Ensure minimum weights (no metric completely ignored)
        min_weight = 0.05
        for key in new_weights:
            if new_weights[key] < min_weight:
                new_weights[key] = min_weight
        
        return new_weights
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of weight adaptations"""
        return self.adaptation_history
    
    def reset_to_initial_weights(self) -> None:
        """Reset weights to initial values"""
        self.alpha = self.initial_weights['alpha']
        self.beta = self.initial_weights['beta']
        if self.enable_locality:
            self.gamma = self.initial_weights['gamma']
        self.delta = self.initial_weights['delta']
        self.epsilon = self.initial_weights['epsilon']
        self.adaptation_history = []
        logger.info("Reset weights to initial values")
