#!/usr/bin/env python3
"""
Complete Metrics Extractor for Nsight Systems Databases

Extracts comprehensive performance metrics from NVIDIA Nsight Systems profiling databases
utilizing ALL tuples from ALL tables for complete analysis. Focuses on:
- Complete MPI communication metrics using all OSRT_API events
- Complete thread performance analysis using all SCHED_EVENTS and COMPOSITE_EVENTS
- Complete GPU performance analysis using all CUPTI tables
- Complete memory locality analysis using all memory-related tables
- Complete ENUM table mappings for accurate event classification
- Complete hardware configuration analysis using TARGET_INFO_SYSTEM_ENV
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json
from dataclasses import dataclass
import time
import re

from autotuner.utils.path_utils import find_slurm_log_for_job


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedMetrics:
    """Container for extracted performance metrics using complete dataset analysis"""
    mpi_comm_time: float = 0.0
    mpip_app_time: float = 0.0
    # max_i MPITime_i / (AppTime_i + MPITime_i) from mpiP — better α than max(MPI)/max(App)
    mpip_max_mpi_wall_fraction: float = 0.0
    total_runtime: float = 0.0
    cpu_utilization: float = 0.0
    memory_local_accesses: int = 0
    memory_total_accesses: int = 0
    numa_efficiency: float = 0.0
    locality_source: str = "missing"
    thread_stall_time: float = 0.0
    load_imbalance: float = 0.0
    cache_miss_rate: float = 0.0
    mpi_call_count: int = 0
    mpi_message_size_total: int = 0
    thread_context_switches: int = 0
    memory_bandwidth: float = 0.0
    
    # Enhanced GPU metrics
    gpu_kernel_time: float = 0.0
    gpu_memcpy_time: float = 0.0
    gpu_sync_time: float = 0.0
    gpu_busy_time: float = 0.0  # kernel+memcpy+sync seconds (for wall-clock-aligned δ)
    gpu_utilization: float = 0.0
    gpu_memory_bandwidth: float = 0.0
    gpu_kernel_count: int = 0
    gpu_memcpy_count: int = 0
    gpu_active_span: float = 0.0
    cupti_process_count: int = 0
    gpu_utilization_source: str = ""
    
    # Enhanced OpenMP metrics
    openmp_work_time: float = 0.0
    openmp_sync_time: float = 0.0
    openmp_task_count: int = 0
    openmp_work_efficiency: float = 0.0
    openmp_instrumented: bool = False
    
    # Enhanced memory metrics
    memory_transfer_rate: float = 0.0
    memory_bottleneck_severity: float = 0.0     
    
    # Complete dataset analysis fields (added dynamically)
    hardware_configuration: Optional[Dict[str, Any]] = None
    enum_mappings: Optional[Dict[str, Dict[int, str]]] = None
    performance_events: Optional[Dict[str, Any]] = None
    
    # Distribution analysis fields (for bottleneck detection)
    distribution_stats: Optional[Dict[str, Any]] = None
    per_rank_breakdown: Optional[Dict[str, Any]] = None
    per_thread_breakdown: Optional[Dict[str, Any]] = None
    per_stream_breakdown: Optional[Dict[str, Any]] = None
    outliers: Optional[Dict[str, List[Any]]] = None
    bottleneck_details: Optional[Dict[str, Any]] = None

class NsightMetricsExtractor:
    """
    Extracts performance metrics from Nsight Systems SQLite databases.
    Processes one .sqlite file at a time; multi-rank Run 2 uses a full-job
    Nsight session (all ranks), so one export contains the merged timeline.

    Analyzes profiling data to extract metrics needed for the heuristic scoring system:
    1. MPI Communication Analysis
    2. Thread Performance Analysis
    3. Memory and NUMA Locality Analysis
    """
    
    def __init__(self, database_path: Path):
        """
        Initialize the extractor with a database path
        
        Args:
            database_path: Path to the Nsight Systems .sqlite file
        """
        self.database_path = Path(database_path)
        self.connection = None
        self.tables_cache = {}
        self._total_runtime_cache: Optional[float] = None

        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
        
        logger.info(f"Initialized metrics extractor for: {self.database_path}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_database()
    
    def connect_database(self) -> None:
        """Connect to the SQLite database"""
        try:
            self.connection = sqlite3.connect(str(self.database_path))
            self.connection.row_factory = sqlite3.Row
            logger.info("Connected to Nsight Systems database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close_database(self) -> None:
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Closed database connection")
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables in the database"""
        if not self.connection:
            self.connect_database()
        
        executor = self.connection.cursor()
        executor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in executor.fetchall()]
        
        logger.info(f"Found {len(tables)} tables in database")
        return tables

    def _get_table_columns(self, table_name: str) -> List[str]:
        """Return list of columns for a given table (empty if missing)"""
        if not self.connection:
            self.connect_database()
        try:
            executor = self.connection.cursor()
            executor.execute(f"PRAGMA table_info({table_name});")
            return [row[1] if isinstance(row, tuple) else row['name'] for row in executor.fetchall()]
        except Exception:
            return []

    def _table_has_columns(self, table_name: str, required: List[str]) -> bool:
        cols = set(self._get_table_columns(table_name))
        return all(col in cols for col in required)

    def _cuda_kernel_activity_tables(self) -> List[str]:
        """
        Candidate SQLite tables for CUDA kernel rows (Nsight export schema varies by version).
        Prefer canonical CUPTI_ACTIVITY_KIND_KERNEL, then any CUPTI*KERNEL* name.
        """
        if not self.connection:
            self.connect_database()
        cur = self.connection.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        names = [row["name"] for row in cur.fetchall()]
        preferred: List[str] = []
        secondary: List[str] = []
        for n in names:
            u = n.upper()
            if u == "CUPTI_ACTIVITY_KIND_KERNEL":
                if n not in preferred:
                    preferred.insert(0, n)
            elif "CUPTI" in u and "KERNEL" in u:
                if n not in preferred:
                    preferred.append(n)
            elif ("CUDA" in u or "GPU" in u) and "KERNEL" in u and "ACTIVITY" in u:
                secondary.append(n)
        seen = set()
        out: List[str] = []
        for n in preferred + secondary:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _kernel_time_column_pair(self, table: str) -> Optional[Tuple[str, str]]:
        cols = set(self._get_table_columns(table))
        for a, b in (
            ("start", "end"),
            ("startTime", "endTime"),
            ("globalStart", "globalEnd"),
            ("timestampStart", "timestampEnd"),
            ("tsBegin", "tsEnd"),
            ("queued", "completed"),
            ("queueStart", "queueEnd"),
        ):
            if a in cols and b in cols:
                return (a, b)
        return None

    def _fetch_kernel_aggregates(self, table: str) -> Optional[Dict[str, Any]]:
        pair = self._kernel_time_column_pair(table)
        if not pair:
            return None
        s, e = pair
        executor = self.connection.cursor()
        q = f"""
        SELECT COUNT(*) AS kernel_count,
               SUM(({e} - {s})/1e9) AS total_kernel_time,
               AVG(({e} - {s})/1e9) AS avg_kernel_time
        FROM {table}
        WHERE {e} > {s}
        """
        try:
            executor.execute(q)
            row = executor.fetchone()
            if not row:
                return None
            kc = row["kernel_count"] or 0
            if kc == 0:
                return None
            return {
                "kernel_count": int(kc),
                "total_kernel_time": float(row["total_kernel_time"] or 0.0),
                "avg_kernel_time": float(row["avg_kernel_time"] or 0.0),
            }
        except Exception as ex:
            logger.debug("Kernel aggregate query failed for %s: %s", table, ex)
            return None

    def _cupti_distinct_process_count(self) -> int:
        """Count distinct CUDA processes in CUPTI tables (multi-rank coverage probe)."""
        if not self.connection:
            self.connect_database()
        pid_cols = ("globalPid", "processId", "pid", "correlationId")
        for table in self._cuda_kernel_activity_tables():
            cols = set(self._get_table_columns(table))
            pid_col = next((c for c in pid_cols if c in cols), None)
            if not pid_col:
                continue
            try:
                cur = self.connection.cursor()
                cur.execute(
                    f"SELECT COUNT(DISTINCT {pid_col}) AS n FROM {table} WHERE {pid_col} IS NOT NULL"
                )
                row = cur.fetchone()
                if row and row["n"]:
                    return int(row["n"])
            except Exception:
                continue
        return 0

    def _gpu_active_span_seconds(self) -> float:
        """Wall span from first to last CUPTI kernel (duty-cycle estimate)."""
        if not self.connection:
            self.connect_database()
        for table in self._cuda_kernel_activity_tables():
            pair = self._kernel_time_column_pair(table)
            if not pair:
                continue
            s, e = pair
            try:
                cur = self.connection.cursor()
                cur.execute(
                    f"SELECT MIN({s}) AS t0, MAX({e}) AS t1 FROM {table} WHERE {e} > {s}"
                )
                row = cur.fetchone()
                if row and row["t0"] is not None and row["t1"] is not None:
                    span = (float(row["t1"]) - float(row["t0"])) / 1e9
                    if span > 0:
                        return span
            except Exception:
                continue
        return 0.0

    def _string_table_names(self) -> List[str]:
        if not self.connection:
            self.connect_database()
        cur = self.connection.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row["name"] for row in cur.fetchall() if "string" in row["name"].lower()]

    def _extract_openmp_from_trace(self) -> Dict[str, Any]:
        """
        ε from NVTX OpenMP ranges and/or libgomp/libomp OSRT symbols when present.
        Falls back to trace-based scheduling estimate via extract_all_metrics caller.
        """
        out = {
            "openmp_work_time": 0.0,
            "openmp_sync_time": 0.0,
            "openmp_task_count": 0,
            "openmp_work_efficiency": 0.0,
            "openmp_instrumented": False,
        }
        if not self.connection:
            self.connect_database()

        work_time = 0.0
        sync_time = 0.0
        task_count = 0
        instrumented = False

        # NVTX user ranges (hybrid_vec with USE_NVTX, or OMPT-adjacent markers)
        for nvtx_table in ("NVTX_EVENTS", "NVTX_PAYLOAD_EVENTS"):
            if nvtx_table not in self.get_available_tables():
                continue
            cols = set(self._get_table_columns(nvtx_table))
            text_col = next((c for c in ("text", "message", "name", "payload") if c in cols), None)
            if not text_col:
                continue
            time_pair = self._kernel_time_column_pair(nvtx_table)
            if not time_pair:
                if {"start", "end"}.issubset(cols):
                    time_pair = ("start", "end")
                else:
                    continue
            s, e = time_pair
            try:
                cur = self.connection.cursor()
                cur.execute(
                    f"SELECT {text_col} AS label, SUM(({e}-{s})/1e9) AS dur, COUNT(*) AS cnt "
                    f"FROM {nvtx_table} WHERE {e} > {s} GROUP BY {text_col}"
                )
                for row in cur.fetchall():
                    label = str(row["label"] or "").lower()
                    dur = float(row["dur"] or 0.0)
                    cnt = int(row["cnt"] or 0)
                    if not label or dur <= 0:
                        continue
                    if any(
                        k in label
                        for k in ("omp_init", "omp_loop", "omp_work", "hybrid_omp")
                    ):
                        work_time += dur
                        task_count += cnt
                        instrumented = True
                    elif any(
                        k in label for k in ("omp_barrier", "omp_sync", "omp_wait")
                    ):
                        sync_time += dur
                        instrumented = True
            except Exception as ex:
                logger.debug("NVTX OpenMP parse failed (%s): %s", nvtx_table, ex)

        # OSRT: libgomp / libomp / Intel OMP runtime symbols
        if "OSRT_API" in self.get_available_tables():
            osrt_cols = set(self._get_table_columns("OSRT_API"))
            if {"start", "end", "nameId"}.issubset(osrt_cols):
                st_tables = self._string_table_names()
                for st in st_tables:
                    st_cols = set(self._get_table_columns(st))
                    id_col = next(
                        (c for c in ("id", "nameId", "valueId") if c in st_cols), None
                    )
                    val_col = next(
                        (c for c in ("value", "name", "demangledName") if c in st_cols),
                        None,
                    )
                    if not id_col or not val_col:
                        continue
                    try:
                        cur = self.connection.cursor()
                        cur.execute(
                            f"""
                            SELECT s.{val_col} AS sym,
                                   SUM((o.end - o.start)/1e9) AS dur,
                                   COUNT(*) AS cnt
                            FROM OSRT_API o
                            JOIN {st} s ON o.nameId = s.{id_col}
                            WHERE o.end > o.start
                            GROUP BY s.{val_col}
                            """
                        )
                        for row in cur.fetchall():
                            sym = str(row["sym"] or "").lower()
                            dur = float(row["dur"] or 0.0)
                            cnt = int(row["cnt"] or 0)
                            if dur <= 0:
                                continue
                            if not any(
                                k in sym
                                for k in (
                                    "gomp",
                                    "kmpc",
                                    "omp_",
                                    "libomp",
                                    "openmp",
                                )
                            ):
                                continue
                            instrumented = True
                            if any(
                                k in sym
                                for k in (
                                    "barrier",
                                    "fork",
                                    "join",
                                    "wait",
                                    "reduce",
                                    "critical",
                                )
                            ):
                                sync_time += dur
                            else:
                                work_time += dur
                            task_count += cnt
                        if instrumented:
                            break
                    except Exception as ex:
                        logger.debug("OSRT OpenMP symbol parse (%s): %s", st, ex)

        total = work_time + sync_time
        if instrumented and total > 1e-9:
            eff = work_time / total
            out.update(
                {
                    "openmp_work_time": work_time,
                    "openmp_sync_time": sync_time,
                    "openmp_task_count": task_count,
                    "openmp_work_efficiency": float(max(0.0, min(1.0, eff))),
                    "openmp_instrumented": True,
                }
            )
            logger.info(
                "OpenMP from trace symbols/NVTX: work=%.3fs sync=%.3fs ε=%.3f",
                work_time,
                sync_time,
                out["openmp_work_efficiency"],
            )
        return out

    def extract_mpi_communication_metrics(self, backend_prefix: str = "MPI_") -> Dict[str, Any]:
        """
        Extract communication metrics from the database (MPI).
        Priority: mpiP report (real PMPI timing) > Nsight OSRT > syscall estimate.
        """
        if not self.connection:
            self.connect_database()
        
        comm_metrics = {
            'total_comm_time': 0.0,
            'mpi_call_count': 0,
            'comm_call_count': 0,
            'message_sizes': [],
            'comm_patterns': {},
            'blocking_calls': 0,
            'non_blocking_calls': 0,
            'backend_detected': 'unknown'
        }
        
        try:
            # Priority 1: mpiP report (real PMPI-level timing, works for shared-memory MPI)
            mpip_data = self._extract_mpip_report()
            if mpip_data.get('total_comm_time', 0.0) > 0:
                comm_metrics.update(mpip_data)
                comm_metrics['backend_detected'] = 'mpiP'
                logger.info(f"Extracted communication metrics (mpiP): "
                           f"{comm_metrics['comm_call_count']} calls, "
                           f"{comm_metrics['total_comm_time']:.3f}s total time (max-rank)")
                return comm_metrics

            # Priority 2: Nsight OSRT_API (works when MPI uses kernel network calls)
            if 'OSRT_API' in self.get_available_tables():
                comm_metrics.update(self._extract_from_osrt_api(backend_prefix))
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in self.get_available_tables():
                comm_metrics.update(self._extract_memcpy_metrics())
            if 'COMPOSITE_EVENTS' in self.get_available_tables():
                comp_cols = set(self._get_table_columns('COMPOSITE_EVENTS'))
                if ('name' in comp_cols and 'duration' in comp_cols) or (
                    'name' in comp_cols and {'start', 'end'}.issubset(comp_cols)
                ):
                    comm_metrics.update(self._extract_composite_events(backend_prefix))

            if comm_metrics.get('comm_call_count', 0) > 0:
                patterns = comm_metrics.get('comm_patterns', {})
                if any('mpi' in str(k).lower() for k in patterns.keys()):
                    comm_metrics['backend_detected'] = 'MPI'

            if 'comm_call_count' in comm_metrics:
                comm_metrics['mpi_call_count'] = comm_metrics['comm_call_count']

            # Cap comm time at wall-clock when not from mpiP (OSRT/syscall can sum across threads and exceed runtime)
            if comm_metrics.get('backend_detected') != 'mpiP' and comm_metrics.get('total_comm_time', 0) > 0:
                rt = self._get_total_runtime()
                if rt > 0 and comm_metrics['total_comm_time'] > rt:
                    comm_metrics['total_comm_time'] = rt
                    logger.debug(f"Capped total_comm_time at wall-clock runtime {rt:.2f}s for α accuracy")

            logger.info(f"Extracted communication metrics ({comm_metrics.get('backend_detected', 'unknown')}): "
                       f"{comm_metrics['comm_call_count']} calls, "
                       f"{comm_metrics['total_comm_time']:.2f}s total time")

        except Exception as e:
            logger.error(f"Error extracting communication metrics: {e}")
        
        return comm_metrics

    def _extract_mpip_report(self) -> Dict[str, Any]:
        """
        Parse a mpiP report file to get authoritative MPI communication time.
        mpiP wraps every MPI_* call via PMPI, capturing intra-node shared-memory
        MPI that Nsight OSRT cannot see.

        Looks for report files in: <experiment_dir>/mpiP/<job_id>/*.mpiP
        Returns per-rank timing: max_rank_mpi_time is used as mpi_comm_time
        so the scoring reflects the bottleneck rank.
        """
        result = {
            'total_comm_time': 0.0,
            'comm_call_count': 0,
            'max_rank_app_time': 0.0,
            'mpip_max_mpi_wall_fraction': 0.0,
        }
        try:
            db_dir = self.database_path.parent.resolve()
            job_id_match = re.search(r'(\d+)', db_dir.name)
            if job_id_match:
                job_id = job_id_match.group(1)
                experiment_dir = db_dir.parent.parent
            elif db_dir.name == "profiling" and db_dir.parent.name.isdigit():
                job_id = db_dir.parent.name
                experiment_dir = db_dir.parent.parent.parent
            else:
                job_id = db_dir.name
                experiment_dir = db_dir.parent.parent
            experiment_dir = Path(experiment_dir).resolve()
            loaded = NsightMetricsExtractor.load_mpip_comm_metrics_from_experiment(
                experiment_dir, str(job_id)
            )
            result.update(loaded)
        except Exception as e:
            logger.warning(f"mpiP report path resolution failed: {e}")
        return result

    @staticmethod
    def _collect_mpip_report_files(exp_root: Path, job_id: str) -> List[Path]:
        """
        Find mpiP report files for a SLURM job.

        Search order matches result_collector layout (scratch mpiP/<id>/report and
        archived results/<id>/mpiP/report).
        """
        search_dirs = [
            exp_root / "mpiP" / str(job_id) / "report",
            exp_root / "mpiP" / str(job_id),
            exp_root / "results" / str(job_id) / "mpiP" / "report",
            exp_root / "results" / str(job_id) / "mpiP",
        ]
        for directory in search_dirs:
            if directory.is_dir():
                files = sorted(directory.glob("*.mpiP"))
                if files:
                    return files
        legacy = exp_root / "mpiP"
        if legacy.is_dir():
            files = sorted(legacy.glob("*.mpiP"))
            if files:
                return files
        return []

    @staticmethod
    def load_mpip_comm_metrics_from_experiment(experiment_dir: Path, job_id: str) -> Dict[str, Any]:
        """
        Load mpiP communication metrics without an Nsight SQLite file (e.g. Phase-1 jobs:
        Run 1b/1c with mpiP, no Run 2). Used by generate_results and by _extract_mpip_report.
        """
        result = {
            'total_comm_time': 0.0,
            'comm_call_count': 0,
            'max_rank_app_time': 0.0,
            'mpip_max_mpi_wall_fraction': 0.0,
        }
        try:
            from autotuner.utils.path_utils import experiment_path_roots

            exp_roots = experiment_path_roots(Path(experiment_dir))
            mpip_files: List[Path] = []
            for exp_root in exp_roots:
                mpip_files = NsightMetricsExtractor._collect_mpip_report_files(exp_root, str(job_id))
                if mpip_files:
                    break

            if not mpip_files:
                slurm_out = find_slurm_log_for_job(Path(experiment_dir), str(job_id))
                if slurm_out is None or not slurm_out.exists():
                    logger.info(
                        f"mpiP: no .mpiP files; SLURM log not found for job_id={job_id} "
                        f"(searched experiment roots — ensure slurm-<id>.out/.err exist or mpiP under mpiP/<id>/report/)"
                    )
                    return result
                parsed = NsightMetricsExtractor._parse_mpip_from_slurm_output(slurm_out)
                if parsed.get('total_comm_time', 0) > 0:
                    parsed['backend_detected'] = 'mpiP'
                    logger.info(
                        f"mpiP from SLURM stdout ({slurm_out.name}): max_rank_mpi={parsed['total_comm_time']:.3f}s"
                    )
                    return parsed
                logger.debug(
                    f"mpiP: no .mpiP files; {slurm_out.name} has no MPI Time table (mpiP may be file-only to mpiP/<job>/report/)"
                )
                return result

            report_file = mpip_files[0]
            logger.info(f"Found mpiP report: {report_file}")

            per_rank_mpi = []
            per_rank_app = []
            total_calls = 0
            in_mpi_time_section = False

            with open(report_file, 'r') as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    content = raw[1:].strip() if raw.startswith('@') else raw

                    if 'MPI Time' in content:
                        in_mpi_time_section = True
                        continue
                    if in_mpi_time_section and ('Aggregate Time' in content or 'Callsite' in content):
                        in_mpi_time_section = False

                    if in_mpi_time_section:
                        parts = content.split()
                        if len(parts) >= 3:
                            try:
                                rank_str = parts[0]
                                app_t = float(parts[1])
                                mpi_t = float(parts[2])
                                if rank_str == '*':
                                    continue
                                per_rank_mpi.append(mpi_t)
                                per_rank_app.append(app_t)
                            except ValueError:
                                pass

                    if 'MPI_' in content and content.startswith('MPI_'):
                        parts = content.split()
                        if len(parts) >= 4:
                            try:
                                total_calls += int(parts[3])
                            except (ValueError, IndexError):
                                pass

            if per_rank_mpi:
                # Sanity check: in the mpiP MPI-Time table the assumed column order is
                # "Task  AppTime  MPITime  MPI%" (parts[1]=AppTime, parts[2]=MPITime).
                # Some mpiP builds / wrapper scripts swap these two columns.
                # Heuristic: AppTime must be >= MPITime for every rank (MPI time is a
                # sub-component of total application time).  If the majority of rank
                # pairs violate this, the columns were swapped — correct them.
                if per_rank_app and len(per_rank_mpi) == len(per_rank_app):
                    swapped_count = sum(
                        1 for m, a in zip(per_rank_mpi, per_rank_app) if m > a * 1.05
                    )
                    if swapped_count > len(per_rank_mpi) // 2:
                        logger.warning(
                            f"mpiP column order appears swapped "
                            f"({swapped_count}/{len(per_rank_mpi)} ranks have MPITime > AppTime). "
                            f"Correcting AppTime↔MPITime assignment."
                        )
                        per_rank_mpi, per_rank_app = per_rank_app, per_rank_mpi

                result['total_comm_time'] = max(per_rank_mpi)
                result['max_rank_mpi_time'] = max(per_rank_mpi)
                result['avg_rank_mpi_time'] = sum(per_rank_mpi) / len(per_rank_mpi)
                result['max_rank_app_time'] = max(per_rank_app) if per_rank_app else 0.0
                result['num_ranks_profiled'] = len(per_rank_mpi)
                result['comm_call_count'] = total_calls
                result['mpi_call_count'] = total_calls
                if len(per_rank_mpi) == len(per_rank_app) and per_rank_app:
                    wall_fracs = [
                        m / max(m + a, 1e-9) for m, a in zip(per_rank_mpi, per_rank_app)
                    ]
                    result['mpip_max_mpi_wall_fraction'] = float(min(1.0, max(wall_fracs)))
                logger.info(
                    f"mpiP: {len(per_rank_mpi)} ranks, "
                    f"max_rank_app={result['max_rank_app_time']:.3f}s, "
                    f"max_rank_mpi={max(per_rank_mpi):.3f}s, "
                    f"avg_mpi={result['avg_rank_mpi_time']:.3f}s, "
                    f"max_mpi_wall_frac={result.get('mpip_max_mpi_wall_fraction', 0.0):.3f}"
                )
        except Exception as e:
            logger.warning(f"mpiP report parsing failed: {e}")

        return result

    @staticmethod
    def _parse_mpip_from_slurm_output(slurm_out: Path) -> Dict[str, Any]:
        """
        Parse mpiP MPI Time section from SLURM stdout when .mpiP files were not written
        (e.g. "Single collector task" or "Could not open ... .mpiP, writing to stdout").
        Handles: "@--- MPI Time (seconds) ---", table "Task AppTime MPITime MPI%", then rows like "0  32.1  6.19  19.28".
        """
        result = {
            'total_comm_time': 0.0,
            'comm_call_count': 0,
            'max_rank_app_time': 0.0,
            'mpip_max_mpi_wall_fraction': 0.0,
        }
        section_markers = [
            'MPI Time (seconds)', 'MPI Time(seconds)', 'MPI Time (sec)', 'MPI Time(sec)',
            'MPI Time', 'Aggregate Time', '@ MPI Time', '@--- MPI Time',
        ]
        try:
            text = slurm_out.read_text(encoding='utf-8', errors='replace')
            lines = text.splitlines()
            in_section = False
            per_rank_mpi = []
            per_rank_app = []
            for line in lines:
                if any(m in line for m in section_markers):
                    in_section = True
                    continue
                if in_section:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('-'):
                        continue
                    # Skip header line (Task AppTime MPITime MPI%)
                    if 'Task' in line or ('AppTime' in line and 'MPITime' in line) or 'MPI%' in line:
                        continue
                    # mpiP table: rank  AppTime  MPITime  MPI%  (space- or tab-separated)
                    parts = re.split(r'[\s\t,]+', stripped)
                    parts = [p for p in parts if p]
                    if len(parts) >= 3:
                        try:
                            rank_str = parts[0].strip()
                            if rank_str == '*':
                                break
                            floats = []
                            for p in parts[1:]:
                                try:
                                    floats.append(float(p))
                                except ValueError:
                                    pass
                            if len(floats) >= 2:
                                # Column order: AppTime, MPITime, [MPI%]; MPITime is index 1
                                app_t = floats[0]
                                mpi_t = floats[1]
                                per_rank_app.append(app_t)
                                per_rank_mpi.append(mpi_t)
                        except (ValueError, IndexError):
                            pass
                    if stripped.startswith('---') and per_rank_mpi:
                        break
            if per_rank_mpi:
                if per_rank_app and len(per_rank_mpi) == len(per_rank_app):
                    swapped_count = sum(
                        1 for m, a in zip(per_rank_mpi, per_rank_app) if m > a * 1.05
                    )
                    if swapped_count > len(per_rank_mpi) // 2:
                        logger.warning(
                            f"mpiP SLURM-stdout: column order appears swapped "
                            f"({swapped_count}/{len(per_rank_mpi)} ranks MPITime > AppTime). "
                            f"Correcting AppTime↔MPITime."
                        )
                        per_rank_mpi, per_rank_app = per_rank_app, per_rank_mpi
                result['total_comm_time'] = max(per_rank_mpi)
                result['max_rank_mpi_time'] = max(per_rank_mpi)
                result['max_rank_app_time'] = max(per_rank_app) if per_rank_app else 0.0
                result['comm_call_count'] = 0
                if len(per_rank_mpi) == len(per_rank_app) and per_rank_app:
                    wall_fracs = [
                        m / max(m + a, 1e-9) for m, a in zip(per_rank_mpi, per_rank_app)
                    ]
                    result['mpip_max_mpi_wall_fraction'] = float(min(1.0, max(wall_fracs)))
        except Exception as e:
            logger.debug(f"Parse mpiP from SLURM output failed: {e}")
        return result

    
    def _extract_from_osrt_api(self, backend_prefix: str = "MPI_") -> Dict[str, Any]:
        """Extract communication metrics from OSRT_API table (MPI)"""
        executor = self.connection.cursor()

        # Prefer joining nameId -> StringIds.value when available
        has_stringids = 'StringIds' in self.get_available_tables()
        osrt_cols = set(self._get_table_columns('OSRT_API'))

        # Duration is end-start in nanoseconds; convert to seconds
        if {'start', 'end', 'nameId'}.issubset(osrt_cols):
            if has_stringids and {'id', 'value'}.issubset(set(self._get_table_columns('StringIds'))):
                # Group by globalTid to prevent concurrently overlapping thread calls from summing greater than wall-clock time
                if 'globalTid' in osrt_cols:
                    comm_query = """
                    SELECT 
                        SUM(call_count) AS call_count,
                        MAX(thread_duration) AS total_duration
                    FROM (
                        SELECT 
                            COUNT(*) AS call_count,
                            SUM((o.end - o.start)/1e9) AS thread_duration
                        FROM OSRT_API o
                        LEFT JOIN StringIds s ON o.nameId = s.id
                        WHERE s.value LIKE '%MPI%' OR s.value LIKE '%mpi%' 
                        GROUP BY o.globalTid
                    )
                    """
                else:
                    comm_query = """
                    SELECT 
                        COUNT(*) AS call_count,
                        SUM((o.end - o.start)/1e9) AS total_duration
                    FROM OSRT_API o
                    LEFT JOIN StringIds s ON o.nameId = s.id
                    WHERE s.value LIKE '%MPI%' OR s.value LIKE '%mpi%' 
                    """
                try:
                    executor.execute(comm_query)
                    res = executor.fetchone()
                except Exception:
                    res = None

                results = []
                if res and (res['total_duration'] or 0) > 0:
                    # Determine which backend was detected
                    backend_name = 'COMM'  # Generic
                    if backend_prefix == "MPI_":
                        backend_name = 'MPI'
                    results = [{'name': f'{backend_name}_calls', 'call_count': res['call_count'] or 0, 'avg_duration': 0.0, 'total_duration': res['total_duration'] or 0.0}]
                else:
                    results = []
            else:
                # No StringIds → cannot reliably detect MPI or syscalls by name
                results = []
        else:
            # Unsupported schema
            return {}

        # No MPI names found → syscall estimate fallback without GROUP BY (tight list)
        if not results and has_stringids:
            # Restrict to direct network data path syscalls only
            syscall_names = (
                "'send','sendto','sendmsg','recv','recvfrom','recvmsg',"
                "'accept','accept4','connect','socket','shutdown'"
            )
            if 'globalTid' in osrt_cols:
                fallback_query = f"""
                SELECT 
                    SUM(call_count) AS call_count,
                    MAX(thread_duration) AS total_duration
                FROM (
                    SELECT 
                        COUNT(*) AS call_count,
                        SUM((o.end - o.start)/1e9) AS thread_duration
                    FROM OSRT_API o
                    LEFT JOIN StringIds s ON o.nameId = s.id
                    WHERE s.value IN ({syscall_names})
                    GROUP BY o.globalTid
                )
                """
            else:
                fallback_query = f"""
                SELECT 
                    COUNT(*) AS call_count,
                    SUM((o.end - o.start)/1e9) AS total_duration
                FROM OSRT_API o
                LEFT JOIN StringIds s ON o.nameId = s.id
                WHERE s.value IN ({syscall_names})
                """
            try:
                executor.execute(fallback_query)
                res = executor.fetchone()
                if res and (res['total_duration'] or 0) > 0:
                    results = [{'name': 'syscall_comm_estimate', 'call_count': res['call_count'] or 0, 'avg_duration': 0.0, 'total_duration': res['total_duration'] or 0.0}]
            except Exception:
                results = []

        # If still no results (e.g., no MPI names), estimate communication via syscall proxies
        # using StringIds values commonly associated with network I/O and waiting.
        if not results and has_stringids:
            syscall_filters = [
                "%send%", "%sendto%", "%sendmsg%", "%recv%", "%recvfrom%", "%recvmsg%",
                "%poll%", "%ppoll%", "%epoll_wait%", "%epoll_ctl%", "%select%", "%pselect%",
                "%accept%", "%accept4%", "%connect%", "%socket%", "%shutdown%"
            ]
            where_clause = " OR ".join(["s.value LIKE '" + f + "'" for f in syscall_filters])
            if 'globalTid' in osrt_cols:
                fallback_query = f"""
                SELECT 
                    name,
                    SUM(call_count) AS call_count,
                    MAX(avg_dur) AS avg_duration,
                    MAX(thread_duration) AS total_duration
                FROM (
                    SELECT 
                        s.value AS name,
                        o.globalTid,
                        COUNT(*) AS call_count,
                        AVG((o.end - o.start)/1e9) AS avg_dur,
                        SUM((o.end - o.start)/1e9) AS thread_duration
                    FROM OSRT_API o
                    LEFT JOIN StringIds s ON o.nameId = s.id
                    WHERE {where_clause}
                    GROUP BY s.value, o.globalTid
                )
                GROUP BY name
                """
            else:
                fallback_query = f"""
                SELECT 
                    s.value AS name,
                    COUNT(*) AS call_count,
                    AVG((o.end - o.start)/1e9) AS avg_duration,
                    SUM((o.end - o.start)/1e9) AS total_duration
                FROM OSRT_API o
                LEFT JOIN StringIds s ON o.nameId = s.id
                WHERE {where_clause}
                GROUP BY s.value
                """
            try:
                executor.execute(fallback_query)
                results = executor.fetchall()
            except Exception:
                results = []

        comm_data = {
            'total_comm_time': 0.0,
            'mpi_call_count': 0,  # Backward compatibility
            'comm_call_count': 0,  # Generic name
            'comm_patterns': {},
            'blocking_calls': 0,
            'non_blocking_calls': 0
        }

        for row in results:
            name = row['name'] or 'UNKNOWN'
            call_count = row['call_count'] or 0
            total_duration = row['total_duration'] or 0.0

            comm_data['total_comm_time'] += total_duration
            comm_data['comm_call_count'] += call_count
            comm_data['mpi_call_count'] += call_count  # Backward compatibility

            lower_name = str(name).lower()
            # Detect non-blocking calls for MPI
            is_nonblocking = (
                ('isend' in lower_name or 'irecv' in lower_name or 'nonblocking' in lower_name or 
                 'i' == lower_name[:1] and 'mpi' in lower_name) or
                ('async' in lower_name or 'async_send' in lower_name or 'async_recv' in lower_name)
            )
            
            if is_nonblocking:
                comm_data['non_blocking_calls'] += call_count
            elif 'mpi' in lower_name:
                comm_data['blocking_calls'] += call_count

            # Only include a compact pattern entry to avoid heavy memory usage
            comm_data['comm_patterns'][str(name)] = {
                'call_count': call_count,
                'total_duration': total_duration,
                'avg_duration': (row.get('avg_duration', 0.0) if isinstance(row, dict) else 0.0)
            }

        return comm_data
    
    def _extract_memcpy_metrics(self) -> Dict[str, Any]:
        """Extract memory copy metrics (schema-robust)"""
        executor = self.connection.cursor()

        if 'CUPTI_ACTIVITY_KIND_MEMCPY' not in self.get_available_tables():
            return {}

        cols = set(self._get_table_columns('CUPTI_ACTIVITY_KIND_MEMCPY'))

        # bytes column may be named 'bytes' or 'size'
        bytes_expr = 'bytes' if 'bytes' in cols else ('size' if 'size' in cols else None)
        # duration as end-start
        duration_expr = '(end - start)/1e9' if {'start','end'}.issubset(cols) else None

        if bytes_expr is None and duration_expr is None:
            return {}

        select_parts = ["COUNT(*) as transfer_count"]
        if bytes_expr:
            select_parts.append(f"SUM({bytes_expr}) as total_bytes")
            select_parts.append(f"AVG({bytes_expr}) as avg_size")
        if duration_expr:
            select_parts.append(f"SUM({duration_expr}) as total_duration")

        memcpy_query = f"""
        SELECT 
            {', '.join(select_parts)}
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """

        executor.execute(memcpy_query)
        row = executor.fetchone()

        if not row:
            return {}

        # row keys may be index-based; use dict-like access where possible
        def get_val(key, default=0):
            try:
                return row[key]
            except Exception:
                return default

        return {
            'memcpy_transfers': get_val('transfer_count', 0),
            'total_data_transferred': get_val('total_bytes', 0),
            'avg_transfer_size': get_val('avg_size', 0),
            'memcpy_time': get_val('total_duration', 0.0)
        }
    
    def _extract_composite_events(self, backend_prefix: str = "MPI_") -> Dict[str, Any]:
        """Extract metrics from composite events table (MPI)"""
        executor = self.connection.cursor()
        cols = set(self._get_table_columns('COMPOSITE_EVENTS'))
        if 'name' in cols and 'duration' in cols:
            composite_query = """
            SELECT 
                name,
                COUNT(*) as event_count,
                SUM(duration) as total_duration
            FROM COMPOSITE_EVENTS
            WHERE name LIKE '%MPI%' OR name LIKE '%mpi%' 
               OR name LIKE '%Communication%'
            GROUP BY name
            """
        elif 'name' in cols and {'start','end'}.issubset(cols):
            composite_query = """
            SELECT 
                name,
                COUNT(*) as event_count,
                SUM((end - start)/1e9) as total_duration
            FROM COMPOSITE_EVENTS
            WHERE name LIKE '%MPI%' OR name LIKE '%mpi%' 
               OR name LIKE '%Communication%'
            GROUP BY name
            """
        else:
            return {}

        executor.execute(composite_query)
        results = executor.fetchall()

        composite_data = {}
        for row in results:
            name = row['name']
            composite_data[name] = {
                'event_count': row['event_count'] or 0,
                'total_duration': row['total_duration'] or 0.0
            }

        return {'composite_events': composite_data}
    
    def extract_thread_performance_metrics(self) -> Dict[str, Any]:
        """
        Extract thread performance and stall metrics
        
        Returns:
            Dictionary containing thread performance analysis
        """
        if not self.connection:
            self.connect_database()
        
        thread_metrics = {
            'cpu_utilization': 0.0,
            'thread_stall_time': 0.0,
            'context_switches': 0,
            'load_imbalance': 0.0,
            'thread_count': 0
        }
        
        try:
            # Check for SCHED_EVENTS table (scheduling events)
            if 'SCHED_EVENTS' in self.get_available_tables():
                thread_metrics.update(self._extract_sched_events())
            
            # GPU kernel rows (table/column names vary by Nsight export version)
            km = self._extract_kernel_metrics()
            if km:
                thread_metrics.update(km)
            
            # Fallback: if we still don't have stall time, estimate from OSRT_API blocking syscalls
            if thread_metrics['thread_stall_time'] <= 0 and 'OSRT_API' in self.get_available_tables():
                osrt_stall = self._extract_stall_from_osrt()
                # Normalize stall estimate by total OSRT_API time and scale to total runtime
                stall_estimate_s = osrt_stall.get('stall_time_estimate', 0.0)
                osrt_total = osrt_stall.get('osrt_total_time', 0.0)
                total_time = self._get_total_runtime()
                if stall_estimate_s > 0 and osrt_total > 0 and total_time > 0:
                    normalized_stall = min(max(stall_estimate_s / osrt_total, 0.0), 1.0) * total_time
                    thread_metrics['thread_stall_time'] = normalized_stall
                # Always carry over event counts if present
                if 'total_sched_events' in osrt_stall and thread_metrics.get('total_sched_events', 0) == 0:
                    thread_metrics['total_sched_events'] = osrt_stall['total_sched_events']
            
            # β (thread): Single consistent definition — stall-based first, then COMPOSITE_EVENTS (threadState=1 running), else 0
            if thread_metrics['thread_stall_time'] > 0:
                total_time = self._get_total_runtime()
                if total_time > 0:
                    calculated_util = 1.0 - (thread_metrics['thread_stall_time'] / total_time)
                    thread_metrics['cpu_utilization'] = calculated_util
                    logger.debug(f"CPU utilization from stall time: {calculated_util:.3f} (stall={thread_metrics['thread_stall_time']:.2f}s, total={total_time:.2f}s)")
            else:
                # No stall time: use COMPOSITE_EVENTS (threadState/cpuCycles) when available for β accuracy
                composite_util = self._extract_cpu_util_from_composite_events()
                if composite_util is not None and composite_util > 0:
                    thread_metrics['cpu_utilization'] = composite_util
                    logger.debug(f"CPU utilization from COMPOSITE_EVENTS: {composite_util:.3f}")
                elif 'cpu_utilization' not in thread_metrics or thread_metrics.get('cpu_utilization', 0.0) == 0.0:
                    thread_metrics['cpu_utilization'] = 0.0
                    logger.debug("No thread stall or COMPOSITE_EVENTS data, CPU utilization set to 0.0 (unknown)")
            
            # Clamp into bounds
            if thread_metrics['cpu_utilization'] < 0:
                thread_metrics['cpu_utilization'] = 0.0
            if thread_metrics['cpu_utilization'] > 1:
                thread_metrics['cpu_utilization'] = 1.0

            # Load imbalance from per-thread running cycles (input to trace-derived OpenMP ε)
            load_imb = self._extract_load_imbalance_from_composite_events()
            if load_imb is not None:
                thread_metrics['load_imbalance'] = load_imb
            
            logger.info(f"Extracted thread metrics: CPU util={thread_metrics['cpu_utilization']:.3f}, "
                       f"stall time={thread_metrics['thread_stall_time']:.2f}s, load_imbalance={thread_metrics['load_imbalance']:.3f}")
            
        except Exception as e:
            logger.error(f"Error extracting thread metrics: {e}")
        
        return thread_metrics
    
    def _extract_sched_events(self) -> Dict[str, Any]:
        """Extract scheduling events for thread analysis (schema-robust)"""
        executor = self.connection.cursor()
        cols = set(self._get_table_columns('SCHED_EVENTS'))
        if not cols:
            return {}
        # Minimal extraction: total events and distinct threads available in your schema
        sched_query = """
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT globalTid) as thread_count
        FROM SCHED_EVENTS
        """
        executor.execute(sched_query)
        row = executor.fetchone()
        if row:
            return {
                'thread_stall_time': 0.0,
                'thread_count': row['thread_count'] or 0,
                'total_sched_events': row['total_events'] or 0
            }
        return {}

    def _extract_cpu_util_from_composite_events(self) -> Optional[float]:
        """
        Extract CPU utilization from COMPOSITE_EVENTS (threadState + cpuCycles).
        Nsight uses threadState: 1 = running, other states = idle/blocked.
        Returns utilization in [0, 1] or None if not available.
        """
        if 'COMPOSITE_EVENTS' not in self.get_available_tables():
            return None
        cols = set(self._get_table_columns('COMPOSITE_EVENTS'))
        if 'threadState' not in cols or 'cpuCycles' not in cols:
            return None
        try:
            executor = self.connection.cursor()
            executor.execute("""
                SELECT threadState, SUM(cpuCycles) AS total_cycles
                FROM COMPOSITE_EVENTS
                GROUP BY threadState
            """)
            rows = executor.fetchall()
            if not rows:
                return None
            total_cycles = sum(row['total_cycles'] or 0 for row in rows)
            if total_cycles <= 0:
                return None
            # State 1 is typically "running" in Nsight Systems
            running_cycles = 0
            for row in rows:
                if row['threadState'] == 1:
                    running_cycles = row['total_cycles'] or 0
                    break
            util = running_cycles / total_cycles
            util = max(0.0, min(1.0, util))
            logger.debug(f"CPU utilization from COMPOSITE_EVENTS: {util:.3f} (running/total cycles)")
            return util
        except Exception as e:
            logger.debug(f"COMPOSITE_EVENTS CPU util extraction failed: {e}")
            return None

    def _extract_load_imbalance_from_composite_events(self) -> Optional[float]:
        """
        Compute load imbalance from per-thread running cycles (COMPOSITE_EVENTS, threadState=1).
        Imbalance = 1 - (min_running / max_running) across *worker* threads only (ignore threads
        with zero or negligible running cycles so GPU/system threads don't force imbalance=1).
        Returns value in [0, 1] or None if not available.
        """
        if 'COMPOSITE_EVENTS' not in self.get_available_tables():
            return None
        cols = set(self._get_table_columns('COMPOSITE_EVENTS'))
        if 'threadState' not in cols or 'cpuCycles' not in cols:
            return None
        tid_col = 'globalTid' if 'globalTid' in cols else ('tid' if 'tid' in cols else None)
        if not tid_col:
            return None
        try:
            executor = self.connection.cursor()
            executor.execute(f"""
                SELECT {tid_col} AS tid, SUM(cpuCycles) AS running_cycles
                FROM COMPOSITE_EVENTS
                WHERE threadState = 1
                GROUP BY {tid_col}
            """)
            rows = executor.fetchall()
            if not rows or len(rows) < 2:
                return 0.0
            cycles_per_thread = [row['running_cycles'] or 0 for row in rows]
            max_cycles = max(cycles_per_thread)
            if max_cycles <= 0:
                return 0.0
            # Only consider worker threads: running_cycles > 1% of max (exclude idle/GPU/system threads)
            threshold = max_cycles * 0.01
            worker_cycles = [c for c in cycles_per_thread if c > threshold]
            if len(worker_cycles) < 2:
                return 0.0
            min_worker = min(worker_cycles)
            # 1 - min/max: 0 when balanced, <1 when one worker did less work
            imbalance = 1.0 - (min_worker / max_cycles)
            imbalance = max(0.0, min(1.0, imbalance))
            logger.debug(f"Load imbalance from COMPOSITE_EVENTS: {imbalance:.3f} (worker threads={len(worker_cycles)}, min/max running cycles)")
            return imbalance
        except Exception as e:
            logger.debug(f"Load imbalance extraction failed: {e}")
            return None

    def _extract_stall_from_osrt(self) -> Dict[str, Any]:
        """Estimate stall time from blocking/wait syscalls and total OSRT_API time."""
        try:
            if 'StringIds' not in self.get_available_tables():
                return {}
            cols = set(self._get_table_columns('OSRT_API'))
            if not {'start','end','nameId'}.issubset(cols):
                return {}
            # Tight stall list (remove poll/select/epoll_wait proxies)
            stall_names = (
                "'nanosleep','sleep','pthread_cond_wait','pthread_join','sem_wait','pause'"
            )
            executor = self.connection.cursor()
            q = f"""
            SELECT 
                COUNT(*) AS stall_events,
                SUM((o.end - o.start)/1e9) AS stall_time
            FROM OSRT_API o
            LEFT JOIN StringIds s ON o.nameId = s.id
            WHERE s.value IN ({stall_names})
            """
            executor.execute(q)
            row = executor.fetchone()
            stall_time = (row['stall_time'] or 0.0) if row else 0.0
            stall_events = max((row['stall_events'] or 0), 0) if row else 0

            # Get total OSRT_API time to normalize
            executor.execute("SELECT SUM((end - start)/1e9) AS total_time FROM OSRT_API")
            row2 = executor.fetchone()
            osrt_total = (row2['total_time'] or 0.0) if row2 else 0.0

            return {
                'stall_time_estimate': stall_time,
                'osrt_total_time': osrt_total,
                'total_sched_events': stall_events
            }
        except Exception:
            pass
        return {}
    
    def _extract_kernel_metrics(self) -> Dict[str, Any]:
        """Extract GPU kernel execution metrics from any Nsight-export kernel activity table."""
        for t in self._cuda_kernel_activity_tables():
            agg = self._fetch_kernel_aggregates(t)
            if not agg:
                continue
            out: Dict[str, Any] = {
                "gpu_kernel_count": agg["kernel_count"],
                "gpu_kernel_time": agg["total_kernel_time"],
                "avg_kernel_time": agg["avg_kernel_time"],
            }
            pair = self._kernel_time_column_pair(t)
            cols = set(self._get_table_columns(t))
            thread_col = next((c for c in ("tid", "globalTid") if c in cols), None)
            if pair and thread_col:
                s, e = pair
                try:
                    executor = self.connection.cursor()
                    executor.execute(
                        f"SELECT COUNT(DISTINCT {thread_col}) AS active_threads FROM {t} WHERE {e} > {s}"
                    )
                    row = executor.fetchone()
                    if row:
                        out["gpu_active_threads"] = row["active_threads"] or 0
                except Exception:
                    pass
            return out
        return {}
    
    def extract_memory_locality_metrics(self) -> Dict[str, Any]:
        """
        Extract memory and NUMA locality metrics
        
        Returns:
            Dictionary containing memory locality analysis
        """
        if not self.connection:
            self.connect_database()
        
        memory_metrics = {
            'memory_local_accesses': 0,
            'memory_total_accesses': 0,
            'numa_efficiency': 0.0,
            'cache_miss_rate': 0.0,
            'memory_bandwidth': 0.0
        }
        
        try:
            # Check for memory-related tables
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in self.get_available_tables():
                memory_metrics.update(self._extract_memory_activity())
            
            # Extract hardware counter data from LIKWID (preferred) or Nsight Systems
            likwid_metrics = self._extract_hardware_counters()
            if likwid_metrics.get('numa_local_accesses', 0) > 0 or likwid_metrics.get('numa_remote_accesses', 0) > 0 or likwid_metrics.get('memory_bandwidth', 0.0) > 0:
                # Use LIKWID hardware counter data (precise)
                memory_metrics['memory_local_accesses'] = likwid_metrics.get('numa_local_accesses', 0)
                memory_metrics['memory_total_accesses'] = (
                    likwid_metrics.get('numa_local_accesses', 0) + 
                    likwid_metrics.get('numa_remote_accesses', 0)
                )
                memory_metrics['numa_efficiency'] = likwid_metrics.get('numa_efficiency', 0.0)
                memory_metrics['cache_miss_rate'] = likwid_metrics.get('cache_miss_rate', 0.0)
                memory_metrics['memory_bandwidth'] = likwid_metrics.get('memory_bandwidth', 0.0)
                # Carry the source tag so locality_source can distinguish numastat from real MBOX DRAM
                memory_metrics['numa_source'] = likwid_metrics.get('numa_source', 'missing')
                logger.info("Using LIKWID hardware data for Locality Score (Primary)")
            elif 'TARGET_INFO_GPU' in self.get_available_tables():
                # Fallback to Nsight Systems estimates
                memory_metrics.update(likwid_metrics)
                logger.info("Using Nsight Systems estimates for NUMA metrics (fallback)")
            
            # Calculate NUMA efficiency from local/total when we have counts
            if memory_metrics['memory_total_accesses'] > 0:
                memory_metrics['numa_efficiency'] = (
                    memory_metrics['memory_local_accesses'] / memory_metrics['memory_total_accesses']
                )
            
            logger.info(f"Extracted memory metrics: NUMA efficiency={memory_metrics['numa_efficiency']:.3f}, "
                       f"local/total={memory_metrics['memory_local_accesses']}/{memory_metrics['memory_total_accesses']}")
            
        except Exception as e:
            logger.error(f"Error extracting memory metrics: {e}")
        
        return memory_metrics
    
    def _extract_memory_activity(self) -> Dict[str, Any]:
        """Extract memory activity metrics (schema-robust)"""
        executor = self.connection.cursor()
        cols = set(self._get_table_columns('CUPTI_ACTIVITY_KIND_MEMCPY'))
        if not cols:
            return {}
        # Prefer bytes and end-start duration when present
        select_parts = ["COUNT(*) as total_accesses"]
        if 'bytes' in cols:
            select_parts.append("SUM(bytes) as total_bytes")
        if {'start','end'}.issubset(cols):
            select_parts.append("AVG((end - start)/1e9) as avg_duration")
        memory_query = f"""
        SELECT 
            {', '.join(select_parts)}
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """
        executor.execute(memory_query)
        row = executor.fetchone()
        if row:
            total_accesses = row['total_accesses'] or 0
            estimated_local = int(total_accesses * 0.7)
            return {
                'memory_total_accesses': total_accesses,
                'memory_local_accesses': estimated_local,
                'total_memory_bytes': row['total_bytes'] if 'total_bytes' in row.keys() else 0,
                'avg_memory_latency': row['avg_duration'] if 'avg_duration' in row.keys() else 0.0
            }
        return {}
    
    def _extract_hardware_counters(self) -> Dict[str, Any]:
        """
        Extract hardware performance counters from LIKWID output
        
        LIKWID provides precise NUMA hardware counter data including:
        - Local vs. remote memory accesses
        - Cache miss rates
        - Memory bandwidth per NUMA node
        
        Returns:
            Dictionary containing hardware counter metrics from LIKWID
        """
        hardware_metrics = {
            'cache_miss_rate': 0.0,
            'memory_bandwidth': 0.0,
            'numa_local_accesses': 0,
            'numa_remote_accesses': 0,
            'numa_efficiency': 0.0
        }
        
        try:
            # Determine potential LIKWID output paths (must match where job script writes: $EXPERIMENT_DIR/likwid_output/$SLURM_JOB_ID)
            db_dir = self.database_path.parent.resolve()  # e.g. .../experiment_dir/profiling/636405 or .../results/636405/profiling
            job_id_match = re.search(r'(\d+)', db_dir.name)
            if job_id_match:
                job_id = job_id_match.group(1)
                experiment_dir = db_dir.parent.parent  # .../experiment_dir (parent of profiling/)
            elif db_dir.name == "profiling" and db_dir.parent.name.isdigit():
                # Path is results/<job_id>/profiling/profile.sqlite -> job_id from parent, experiment = results.parent
                job_id = db_dir.parent.name
                experiment_dir = db_dir.parent.parent.parent
            else:
                job_id = db_dir.name
                experiment_dir = db_dir.parent.parent
            
            # Check experiment's likwid_output first (where SLURM job writes); optional results/<job_id>/likwid_output copy
            likwid_dirs = [
                experiment_dir / "likwid_output" / job_id,
                Path("likwid_output") / job_id,
                Path.cwd() / "likwid_output" / job_id,
                db_dir,
            ]
            if db_dir.name == "profiling" and db_dir.parent.name.isdigit():
                likwid_dirs.insert(1, db_dir.parent / "likwid_output")
            
            likwid_files = []
            for d in likwid_dirs:
                if not d.exists():
                    continue
                # Single file (single-rank or legacy)
                for pattern in ["likwid_output.csv", "likwid_output.txt"]:
                    for f in d.glob(pattern):
                        likwid_files.append(f)
                # Per-rank files (multi-rank LIKWID run)
                for pattern in ["likwid_*.csv", "likwid_*.txt"]:
                    for f in sorted(d.glob(pattern), key=lambda x: x.name):
                        if f not in likwid_files:
                            likwid_files.append(f)
                if likwid_files:
                    break
            
            if likwid_files:
                from autotuner.core.likwid_profiler import LIKWIDProfiler
                profiler = LIKWIDProfiler()
                all_local = 0
                all_remote = 0
                eff_sum = 0.0
                eff_count = 0
                cache_rates = []
                for likwid_file in likwid_files:
                    m = profiler.parse_likwid_output(likwid_file)
                    all_local += m.local_dram_accesses
                    all_remote += m.remote_dram_accesses
                    if m.numa_efficiency > 0 or (m.local_dram_accesses + m.remote_dram_accesses) > 0:
                        eff_sum += m.numa_efficiency
                        eff_count += 1
                    if m.cache_miss_rate > 0:
                        cache_rates.append(m.cache_miss_rate)
                # Aggregate: NUMA efficiency = sum_local / (sum_local + sum_remote), or average of per-rank
                total_dram = all_local + all_remote
                if total_dram > 0:
                    hardware_metrics['numa_local_accesses'] = all_local
                    hardware_metrics['numa_remote_accesses'] = all_remote
                    hardware_metrics['numa_efficiency'] = all_local / total_dram
                elif eff_count > 0:
                    hardware_metrics['numa_efficiency'] = eff_sum / eff_count
                if cache_rates:
                    hardware_metrics['cache_miss_rate'] = sum(cache_rates) / len(cache_rates)
                bw_max = 0.0
                for likwid_file in likwid_files:
                    m_bw = profiler.parse_likwid_output(likwid_file)
                    bw_max = max(bw_max, float(m_bw.memory_bandwidth or 0.0))
                if bw_max > 0:
                    hardware_metrics['memory_bandwidth'] = bw_max
                # L3/MEM-only LIKWID (no DRAM local/remote rows): older parses left numa_efficiency at 0 while BW was set.
                if hardware_metrics["memory_bandwidth"] > 0 and hardware_metrics["numa_efficiency"] <= 0:
                    from autotuner.core.likwid_profiler import get_locality_for_job

                    loc_eff, _src = get_locality_for_job(Path(experiment_dir), str(job_id))
                    if loc_eff > 0:
                        hardware_metrics["numa_efficiency"] = float(loc_eff)
                        hardware_metrics["numa_source"] = _src  # propagate source label so the numastat fallback below can override it
                    else:
                        hardware_metrics["numa_efficiency"] = min(
                            1.0, float(hardware_metrics["memory_bandwidth"]) / 80.0
                        )
                        hardware_metrics["numa_source"] = "l3_bandwidth_estimate"

                # Fallback: kernel NUMA sysfs snapshot (LocalNode / OtherNode counters).
                # Written by the SLURM script from /sys/devices/system/node/node*/meminfo — no root needed.
                # Use this whenever LIKWID has NOT provided real DRAM local/remote counters
                # (i.e. source is heuristic: l3_estimate, l3_bandwidth_estimate, socket_l3_balance_*, missing).
                _is_heuristic_numa = hardware_metrics.get("numa_source", "missing") != "dram_local_remote"
                if hardware_metrics["numa_efficiency"] <= 0 or _is_heuristic_numa:
                    try:
                        from autotuner.core.likwid_profiler import parse_numa_meminfo
                        _local, _remote, _eff = parse_numa_meminfo(Path(experiment_dir), str(job_id))
                        if _eff > 0:
                            hardware_metrics["numa_efficiency"] = float(_eff)
                            hardware_metrics["numa_local_accesses"] = int(_local)
                            hardware_metrics["numa_remote_accesses"] = int(_remote)
                            hardware_metrics["numa_source"] = "numa_meminfo"
                            logger.info(
                                f"  Locality (γ) from kernel NUMA sysfs: {_eff:.3f} "
                                f"(local={_local:,}, remote={_remote:,})"
                            )
                    except Exception:
                        pass
                logger.info(f"Extracted LIKWID hardware counters from {len(likwid_files)} file(s): "
                           f"NUMA efficiency={hardware_metrics['numa_efficiency']:.3f}, "
                           f"Memory BW={hardware_metrics['memory_bandwidth']:.2f} GB/s")
            else:
                logger.debug(f"No LIKWID output files found for job {job_id}")
                
        except Exception as e:
            logger.warning(f"Error extracting hardware counters: {e}")
        
        return hardware_metrics
    
    def _get_wall_time_from_run_artifacts(self) -> float:
        """
        Genuine application wall time from Run 1a logs next to the profiling DB.
        Nsight OSRT MIN/MAX often spans the whole trace session (e.g. 100s), not the app.
        """
        try:
            db_dir = self.database_path.parent.resolve()
            job_dir = db_dir.parent
            jid = job_dir.name
            if db_dir.name == "profiling" and str(jid).isdigit():
                pass
            elif str(db_dir.name).isdigit() and db_dir.parent.name == "profiling":
                job_dir = db_dir
                jid = db_dir.name
            else:
                return 0.0
            from autotuner.app_registry.stdout_parsing import parse_slurm_file_runtime_throughput

            names = (f"run1_{jid}.out", f"slurm-{jid}.out")
            # Prefer logs under results/<job_id>/; also walk up — sbatch often writes slurm-%j.out to the experiment cwd.
            search_roots: List[Path] = [job_dir]
            p = job_dir
            for _ in range(8):
                p = p.parent
                if p == p.parent:
                    break
                search_roots.append(p)

            seen = set()
            for root in search_roots:
                rp = root.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                for name in names:
                    path = root / name
                    if path.is_file():
                        rt, _ = parse_slurm_file_runtime_throughput(path)
                        if rt and float(rt) > 0:
                            return float(rt)
            return 0.0
        except Exception as e:
            logger.debug("Run 1a wall time parse failed: %s", e)
            return 0.0

    def _get_total_runtime(self) -> float:
        """Application wall-clock time: prefer Run 1a logs; else Nsight OSRT span; never invent 100s."""
        if self._total_runtime_cache is not None:
            return self._total_runtime_cache

        wall = self._get_wall_time_from_run_artifacts()
        if wall > 0:
            self._total_runtime_cache = wall
            return wall

        if not self.connection:
            self.connect_database()

        executor = self.connection.cursor()
        runtime = 0.0

        if "OSRT_API" in self.get_available_tables():
            cols = set(self._get_table_columns("OSRT_API"))
            if {"start", "end"}.issubset(cols):
                try:
                    executor.execute("SELECT MIN(start), MAX(end) FROM OSRT_API")
                    row = executor.fetchone()
                    if row and row[0] and row[1]:
                        runtime = (row[1] - row[0]) / 1e9
                except Exception:
                    pass

        if runtime <= 0 and "COMPOSITE_EVENTS" in self.get_available_tables():
            ccols = set(self._get_table_columns("COMPOSITE_EVENTS"))
            if {"start", "end"}.issubset(ccols):
                try:
                    executor.execute("SELECT MIN(start), MAX(end) FROM COMPOSITE_EVENTS")
                    row = executor.fetchone()
                    if row and row[0] and row[1]:
                        runtime = (row[1] - row[0]) / 1e9
                except Exception:
                    pass

        if runtime <= 0:
            logger.debug(
                "No application wall time from Run 1a or trace tables; metrics using runtime default to 0.0 where needed"
            )

        self._total_runtime_cache = max(0.0, float(runtime))
        return self._total_runtime_cache
    
    def extract_gpu_metrics(self) -> Dict[str, Any]:
        """
        Extract GPU utilization metrics from CUDA profiling data
        
        Returns:
            Dictionary containing GPU performance metrics
        """
        gpu_metrics = {
            'gpu_kernel_time': 0.0,
            'gpu_memcpy_time': 0.0,
            'gpu_sync_time': 0.0,
            'gpu_utilization': 0.0,
            'gpu_memory_bandwidth': 0.0,
            'gpu_kernel_count': 0,
            'gpu_memcpy_count': 0
        }
        
        try:
            if not self.connection:
                self.connect_database()
            executor = self.connection.cursor()

            for t in self._cuda_kernel_activity_tables():
                agg = self._fetch_kernel_aggregates(t)
                if agg:
                    gpu_metrics["gpu_kernel_time"] = agg["total_kernel_time"]
                    gpu_metrics["gpu_kernel_count"] = agg["kernel_count"]
                    break

            # Extract GPU memory copy metrics
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in self.get_available_tables():
                memcpy_query = """
                SELECT 
                    COUNT(*) as memcpy_count,
                    SUM((end - start)/1e9) as total_memcpy_time,
                    SUM(bytes) as total_bytes_transferred,
                    AVG((end - start)/1e9) as avg_memcpy_time
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE end > start AND bytes > 0
                """
                executor.execute(memcpy_query)
                row = executor.fetchone()
                if row and row["total_memcpy_time"] is not None:
                    gpu_metrics["gpu_memcpy_time"] = row["total_memcpy_time"]
                    gpu_metrics["gpu_memcpy_count"] = row["memcpy_count"]
                    if row["total_memcpy_time"] > 0:
                        gpu_metrics["gpu_memory_bandwidth"] = (
                            row["total_bytes_transferred"] / row["total_memcpy_time"] / 1e9
                        )

            if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in self.get_available_tables():
                sync_query = """
                SELECT 
                    COUNT(*) as sync_count,
                    SUM((end - start)/1e9) as total_sync_time
                FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
                WHERE end > start
                """
                executor.execute(sync_query)
                row = executor.fetchone()
                if row and row["total_sync_time"] is not None:
                    gpu_metrics["gpu_sync_time"] = row["total_sync_time"]

            total_gpu_time = (
                gpu_metrics["gpu_kernel_time"]
                + gpu_metrics["gpu_memcpy_time"]
                + gpu_metrics["gpu_sync_time"]
            )
            gpu_metrics["gpu_busy_time"] = total_gpu_time
            gpu_metrics["gpu_active_span"] = self._gpu_active_span_seconds()
            gpu_metrics["cupti_process_count"] = self._cupti_distinct_process_count()
            total_runtime = self._get_total_runtime()
            if total_runtime > 0:
                gpu_metrics["gpu_utilization"] = min(total_gpu_time / total_runtime, 1.0)

            logger.info(
                f"Extracted GPU metrics: {gpu_metrics['gpu_kernel_count']} kernels, "
                f"{gpu_metrics['gpu_memcpy_count']} memcpys, "
                f"utilization={gpu_metrics['gpu_utilization']:.3f}, "
                f"cupti_procs={gpu_metrics['cupti_process_count']}, "
                f"span={gpu_metrics['gpu_active_span']:.3f}s"
            )

        except Exception as e:
            logger.warning(f"Error extracting GPU metrics: {e}")
        
        return gpu_metrics
    
    def extract_openmp_metrics(self) -> Dict[str, Any]:
        """ε from NVTX/OSRT OpenMP symbols when present; else trace-based scheduling in extract_all_metrics."""
        return self._extract_openmp_from_trace()
    
    def extract_enhanced_memory_metrics(self) -> Dict[str, Any]:
        """
        Extract enhanced memory bandwidth and bottleneck analysis
        
        Returns:
            Dictionary containing enhanced memory metrics
        """
        memory_metrics = {
            'memory_transfer_rate': 0.0,
            'memory_bottleneck_severity': 0.0
        }
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                executor = conn.cursor()
                
                # Analyze memory transfer patterns
                if 'CUPTI_ACTIVITY_KIND_MEMCPY' in self.get_available_tables():
                    # Get memory transfer statistics
                    memcpy_query = """
                    SELECT 
                        copyKind,
                        COUNT(*) as transfer_count,
                        SUM(bytes) as total_bytes,
                        SUM((end - start)/1e9) as total_time,
                        AVG(bytes/((end - start)/1e9)) as avg_bandwidth
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    WHERE end > start AND bytes > 0
                    GROUP BY copyKind
                    """
                    executor.execute(memcpy_query)
                    rows = executor.fetchall()
                    
                    total_transfer_time = 0.0
                    total_bytes = 0
                    bandwidths = []
                    
                    for row in rows:
                        if row['total_time'] and row['total_bytes']:
                            total_transfer_time += row['total_time']
                            total_bytes += row['total_bytes']
                            if row['avg_bandwidth']:
                                bandwidths.append(row['avg_bandwidth'])
                    
                    # Calculate overall transfer rate
                    if total_transfer_time > 0:
                        memory_metrics['memory_transfer_rate'] = total_bytes / total_transfer_time / 1e9  # GB/s
                    
                    # Calculate bottleneck severity based on bandwidth variance
                    if len(bandwidths) > 1:
                        bandwidth_std = np.std(bandwidths)
                        bandwidth_mean = np.mean(bandwidths)
                        if bandwidth_mean > 0:
                            memory_metrics['memory_bottleneck_severity'] = min(bandwidth_std / bandwidth_mean, 1.0)
                
                logger.info(f"Enhanced memory metrics: transfer rate={memory_metrics['memory_transfer_rate']:.2f} GB/s, "
                          f"bottleneck severity={memory_metrics['memory_bottleneck_severity']:.3f}")
                
        except Exception as e:
            logger.warning(f"Error extracting enhanced memory metrics: {e}")
        
        return memory_metrics

    def extract_complete_hardware_configuration(self) -> Dict[str, Any]:
        """
        Extract complete hardware configuration using ALL tuples from TARGET_INFO_SYSTEM_ENV
        
        Returns:
            Dictionary containing complete hardware configuration analysis
        """
        hardware_config = {
            'cpu_cores': 0,
            'cpu_speed_mhz': 0,
            'cpu_architecture': '',
            'cpu_description': '',
            'supported_abis': [],
            'supported_64bit_abis': [],
            'cpu_emc_speed_mhz': 0,
            'supports_xmc_clients': False,
            'cpu_info_json': {},
            'nameEnum_mapping': {},
            'hardware_constraints': {}
        }
        
        try:
            if not self.connection:
                self.connect_database()
            
            executor = self.connection.cursor()
            
            # Extract ALL tuples from TARGET_INFO_SYSTEM_ENV
            if 'TARGET_INFO_SYSTEM_ENV' in self.get_available_tables():
                env_query = """
                SELECT nameEnum, name, value 
                FROM TARGET_INFO_SYSTEM_ENV 
                ORDER BY nameEnum, name
                """
                executor.execute(env_query)
                rows = executor.fetchall()
                
                # Process all tuples to build complete hardware configuration
                for row in rows:
                    nameEnum = row['nameEnum']
                    name = row['name']
                    value = row['value']
                    
                    # Store nameEnum mapping
                    hardware_config['nameEnum_mapping'][nameEnum] = name
                    
                    # Extract specific hardware information based on nameEnum
                    if nameEnum == 1 and name == 'CpuCores':
                        hardware_config['cpu_cores'] = int(value) if value.isdigit() else 0
                    elif nameEnum == 2 and name == 'CpuSpeedMhz':
                        hardware_config['cpu_speed_mhz'] = int(value) if value.isdigit() else 0
                    elif nameEnum == 3 and name == 'CpuArchitecture':
                        hardware_config['cpu_architecture'] = value
                    elif nameEnum == 5 and name == 'SupportedABIs':
                        hardware_config['supported_abis'] = [abi.strip() for abi in value.split(',') if abi.strip()]
                    elif nameEnum == 7 and name == 'Supported64BitABIs':
                        hardware_config['supported_64bit_abis'] = [abi.strip() for abi in value.split(',') if abi.strip()]
                    elif nameEnum == 11 and name == 'CNTFRQMHz':
                        hardware_config['counter_frequency_mhz'] = int(value) if value.isdigit() else 0
                    elif nameEnum == 13 and name == 'CpuDescription':
                        hardware_config['cpu_description'] = value
                    elif nameEnum == 14 and name == 'CpuInfo':
                        try:
                            hardware_config['cpu_info_json'] = json.loads(value)
                        except json.JSONDecodeError:
                            hardware_config['cpu_info_json'] = {}
                    elif nameEnum == 101 and name == 'CpuEmcSpeedMhz':
                        hardware_config['cpu_emc_speed_mhz'] = int(value) if value.isdigit() else 0
                    elif nameEnum == 153 and name == 'SupportsXmcClients':
                        hardware_config['supports_xmc_clients'] = bool(int(value)) if value.isdigit() else False
                
                # Build hardware constraints for auto-tuning
                hardware_config['hardware_constraints'] = {
                    'max_threads': hardware_config['cpu_cores'],
                    'max_mpi_ranks': hardware_config['cpu_cores'],  # Can be optimized based on NUMA
                    'cpu_frequency_ghz': hardware_config['cpu_speed_mhz'] / 1000.0,
                    'architecture': hardware_config['cpu_architecture'],
                    'available_configurations': self._generate_hardware_aware_configs(hardware_config)
                }
                
                logger.info(f"Complete hardware configuration extracted: {hardware_config['cpu_cores']} cores, "
                          f"{hardware_config['cpu_speed_mhz']} MHz, {hardware_config['cpu_architecture']}")
            
        except Exception as e:
            logger.error(f"Error extracting complete hardware configuration: {e}")
        
        return hardware_config
    
    def _generate_hardware_aware_configs(self, hardware_config: Dict[str, Any]) -> List[Dict[str, int]]:
        """
        Generate MPI+OpenMP configurations based on complete hardware analysis
        
        Args:
            hardware_config: Complete hardware configuration
            
        Returns:
            List of optimal MPI+OpenMP configurations
        """
        cpu_cores = hardware_config['cpu_cores']
        if cpu_cores <= 0:
            return []
        
        configs = []
        
        # Configuration 1: Single MPI rank, all OpenMP threads
        configs.append({
            'mpi_ranks': 1,
            'openmp_threads': cpu_cores,
            'description': 'Single MPI rank with all CPU cores',
            'suitability': 'High for embarrassingly parallel workloads'
        })
        
        # Configuration 2: Balanced configurations
        for mpi_ranks in [2, 4, 8, 16]:
            if cpu_cores % mpi_ranks == 0:
                openmp_threads = cpu_cores // mpi_ranks
                configs.append({
                    'mpi_ranks': mpi_ranks,
                    'openmp_threads': openmp_threads,
                    'description': f'Balanced {mpi_ranks}x{openmp_threads} configuration',
                    'suitability': 'Good for balanced workloads'
                })
        
        # Configuration 3: NUMA-aware configurations (if we can detect NUMA domains)
        # This would be enhanced with actual NUMA detection
        if cpu_cores >= 32:  # Assume NUMA for high-core systems
            numa_domains = max(2, cpu_cores // 32)  # Estimate NUMA domains
            for mpi_ranks in [numa_domains, numa_domains * 2]:
                if cpu_cores % mpi_ranks == 0:
                    openmp_threads = cpu_cores // mpi_ranks
                    configs.append({
                        'mpi_ranks': mpi_ranks,
                        'openmp_threads': openmp_threads,
                        'description': f'NUMA-aware {mpi_ranks}x{openmp_threads} configuration',
                        'suitability': 'Optimal for NUMA-aware workloads'
                    })
        
        return configs
    
    def extract_complete_enum_mappings(self) -> Dict[str, Dict[int, str]]:
        """
        Extract complete ENUM table mappings using ALL tuples from ALL ENUM tables
        
        Returns:
            Dictionary mapping ENUM table names to their complete ID->name mappings
        """
        enum_mappings = {}
        
        try:
            if not self.connection:
                self.connect_database()
            
            executor = self.connection.cursor()
            
            # Get all ENUM tables
            executor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ENUM_%'")
            enum_tables = [row['name'] for row in executor.fetchall()]
            
            logger.info(f"Found {len(enum_tables)} ENUM tables")
            
            # Extract mappings from each ENUM table
            for table_name in enum_tables:
                try:
                    # Get table columns
                    executor.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in executor.fetchall()]
                    
                    # Extract ID and name mappings
                    if 'id' in columns and 'name' in columns:
                        enum_query = f"SELECT id, name FROM {table_name} ORDER BY id"
                        executor.execute(enum_query)
                        rows = executor.fetchall()
                        
                        enum_mappings[table_name] = {row['id']: row['name'] for row in rows}
                        logger.info(f"Extracted {len(enum_mappings[table_name])} mappings from {table_name}")
                    
                except Exception as e:
                    logger.warning(f"Error extracting mappings from {table_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error extracting complete ENUM mappings: {e}")
        
        return enum_mappings
    
    def extract_complete_performance_events(self) -> Dict[str, Any]:
        """
        Extract complete performance event analysis using ALL tuples from ALL event tables
        
        Returns:
            Dictionary containing complete performance event analysis
        """
        performance_events = {
            'osrt_api_events': {},
            'cuda_kernel_events': {},
            'cuda_memcpy_events': {},
            'cuda_sync_events': {},
            'sched_events': {},
            'composite_events': {},
            'nvtx_events': {},
            'event_classification': {},
            'performance_patterns': {}
        }
        
        try:
            if not self.connection:
                self.connect_database()
            
            executor = self.connection.cursor()
            
            # Extract ALL OSRT_API events with complete classification
            if 'OSRT_API' in self.get_available_tables():
                osrt_query = """
                SELECT 
                    eventClass,
                    nameId,
                    COUNT(*) as event_count,
                    AVG((end - start)/1e9) as avg_duration,
                    SUM((end - start)/1e9) as total_duration,
                    MIN(start) as first_event,
                    MAX(end) as last_event
                FROM OSRT_API 
                GROUP BY eventClass, nameId
                ORDER BY total_duration DESC
                """
                executor.execute(osrt_query)
                rows = executor.fetchall()
                
                performance_events['osrt_api_events'] = {
                    'total_events': sum(row['event_count'] for row in rows),
                    'total_duration': sum(row['total_duration'] for row in rows),
                    'event_classes': {},
                    'top_events': []
                }
                
                # Classify events by eventClass
                for row in rows:
                    event_class = row['eventClass']
                    if event_class not in performance_events['osrt_api_events']['event_classes']:
                        performance_events['osrt_api_events']['event_classes'][event_class] = {
                            'event_count': 0,
                            'total_duration': 0.0,
                            'events': []
                        }
                    
                    performance_events['osrt_api_events']['event_classes'][event_class]['event_count'] += row['event_count']
                    performance_events['osrt_api_events']['event_classes'][event_class]['total_duration'] += row['total_duration']
                    performance_events['osrt_api_events']['event_classes'][event_class]['events'].append({
                        'nameId': row['nameId'],
                        'event_count': row['event_count'],
                        'avg_duration': row['avg_duration'],
                        'total_duration': row['total_duration']
                    })
                
                # Get top events by duration
                performance_events['osrt_api_events']['top_events'] = sorted(
                    rows, key=lambda x: x['total_duration'], reverse=True
                )[:10]
            
            # Extract ALL CUDA kernel events (table / time columns vary by Nsight export)
            for kt in self._cuda_kernel_activity_tables():
                agg = self._fetch_kernel_aggregates(kt)
                if not agg:
                    continue
                kernel_evt: Dict[str, Any] = {
                    "kernel_count": agg["kernel_count"],
                    "total_kernel_time": agg["total_kernel_time"],
                    "avg_kernel_time": agg["avg_kernel_time"],
                }
                pair = self._kernel_time_column_pair(kt)
                kcols = set(self._get_table_columns(kt))
                if pair and {
                    "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                    "staticSharedMemory", "dynamicSharedMemory",
                }.issubset(kcols):
                    s, e = pair
                    try:
                        extq = f"""
                        SELECT AVG(gridX * gridY * gridZ) AS avg_grid_size,
                               AVG(blockX * blockY * blockZ) AS avg_block_size,
                               SUM(staticSharedMemory) AS total_static_shared_memory,
                               SUM(dynamicSharedMemory) AS total_dynamic_shared_memory
                        FROM {kt}
                        WHERE {e} > {s}
                        """
                        executor.execute(extq)
                        er = executor.fetchone()
                        if er:
                            for fld in (
                                "avg_grid_size",
                                "avg_block_size",
                                "total_static_shared_memory",
                                "total_dynamic_shared_memory",
                            ):
                                if fld in er.keys() and er[fld] is not None:
                                    kernel_evt[fld] = er[fld]
                    except Exception:
                        pass
                performance_events["cuda_kernel_events"] = kernel_evt
                break

            # Extract ALL CUDA memory copy events
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in self.get_available_tables():
                memcpy_query = """
                SELECT 
                    copyKind,
                    COUNT(*) as transfer_count,
                    SUM(bytes) as total_bytes,
                    SUM((end - start)/1e9) as total_time,
                    AVG(bytes/((end - start)/1e9)) as avg_bandwidth
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                GROUP BY copyKind
                ORDER BY total_bytes DESC
                """
                executor.execute(memcpy_query)
                rows = executor.fetchall()
                
                performance_events['cuda_memcpy_events'] = {
                    'total_transfers': sum(row['transfer_count'] for row in rows),
                    'total_bytes': sum(row['total_bytes'] for row in rows),
                    'total_time': sum(row['total_time'] for row in rows),
                    'copy_kinds': {row['copyKind']: {
                        'transfer_count': row['transfer_count'],
                        'total_bytes': row['total_bytes'],
                        'total_time': row['total_time'],
                        'avg_bandwidth': row['avg_bandwidth']
                    } for row in rows}
                }
            
            # Extract ALL scheduling events
            if 'SCHED_EVENTS' in self.get_available_tables():
                sched_query = """
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT globalTid) as unique_threads,
                    COUNT(DISTINCT cpu) as unique_cpus,
                    SUM(isSchedIn) as sched_in_events,
                    COUNT(*) - SUM(isSchedIn) as sched_out_events
                FROM SCHED_EVENTS
                """
                executor.execute(sched_query)
                row = executor.fetchone()
                
                if row:
                    performance_events['sched_events'] = {
                        'total_events': row['total_events'],
                        'unique_threads': row['unique_threads'],
                        'unique_cpus': row['unique_cpus'],
                        'sched_in_events': row['sched_in_events'],
                        'sched_out_events': row['sched_out_events'],
                        'context_switch_rate': row['total_events'] / max(row['unique_threads'], 1)
                    }
            
            # Extract ALL composite events
            if 'COMPOSITE_EVENTS' in self.get_available_tables():
                composite_query = """
                SELECT 
                    threadState,
                    COUNT(*) as event_count,
                    SUM(cpuCycles) as total_cycles,
                    COUNT(DISTINCT globalTid) as unique_threads
                FROM COMPOSITE_EVENTS
                GROUP BY threadState
                ORDER BY event_count DESC
                """
                executor.execute(composite_query)
                rows = executor.fetchall()
                
                performance_events['composite_events'] = {
                    'total_events': sum(row['event_count'] for row in rows),
                    'total_cycles': sum(row['total_cycles'] for row in rows),
                    'thread_states': {row['threadState']: {
                        'event_count': row['event_count'],
                        'total_cycles': row['total_cycles'],
                        'unique_threads': row['unique_threads']
                    } for row in rows}
                }
            
            logger.info(f"Complete performance events extracted: {performance_events['osrt_api_events'].get('total_events', 0)} OSRT events, "
                       f"{performance_events['cuda_kernel_events'].get('kernel_count', 0)} CUDA kernels")
            
        except Exception as e:
            logger.error(f"Error extracting complete performance events: {e}")
        
        return performance_events

    def _openmp_work_efficiency_from_trace(
        self,
        thread_metrics: Dict[str, Any],
        total_runtime: float,
        sched_events: Dict[str, Any],
        final_cpu_utilization: float,
        mpi_comm_fraction: float = 0.0,
    ) -> float:
        """
        OpenMP-related efficiency ε from Nsight trace: stall vs wall, CPU util,
        scheduling churn, load imbalance (harmonic blend of work_frac and util).
        """
        rt = max(float(total_runtime or 0.0), 1e-6)
        stall = max(float(thread_metrics.get("thread_stall_time") or 0.0), 0.0)
        work_frac = max(0.0, min(1.0, 1.0 - min(stall / rt, 1.0)))
        # MPI wait in the Nsight window is not OpenMP parallelism — de-weight when comm dominates.
        mpi_frac = max(0.0, min(1.0, float(mpi_comm_fraction or 0.0)))
        if mpi_frac > 0.05:
            work_frac *= max(0.05, 1.0 - 0.90 * mpi_frac)
        util = max(0.0, min(1.0, float(final_cpu_utilization or 0.0)))

        if work_frac > 1e-6 and util > 1e-6:
            h_blend = 2.0 * work_frac * util / (work_frac + util)
        else:
            h_blend = max(work_frac, util)

        te = int(sched_events.get("total_events") or 0)
        ut = max(int(sched_events.get("unique_threads") or 0), 1)
        churn = float(te) / (rt * float(ut))
        # Typical traces: churn ~1e3–1e4/s/thread is healthy; very high → contention / oversubscription
        sched_smooth = 1.0 / (1.0 + churn / 6000.0)

        load_imb = float(thread_metrics.get("load_imbalance") or 0.0)
        load_imb = min(1.0, max(0.0, load_imb))
        if load_imb >= 0.99:
            imb_f = 1.0
        else:
            imb_f = max(0.08, 1.0 - 0.45 * load_imb)

        eff = h_blend * sched_smooth * imb_f
        return float(max(0.0, min(1.0, eff)))

    def extract_all_metrics(self, backend_prefix: str = "MPI_") -> ExtractedMetrics:
        """
        Extract ALL performance metrics using complete dataset analysis
        
        Utilizes ALL tuples from ALL tables for comprehensive analysis:
        - Complete hardware configuration from TARGET_INFO_SYSTEM_ENV
        - Complete ENUM mappings from all ENUM tables
        - Complete performance events from all event tables
        - Complete GPU analysis from all CUPTI tables
        - Complete communication analysis from all OSRT tables (MPI)
        
        Args:
            backend_prefix: Prefix for communication events ("MPI_")
        
        Returns:
            ExtractedMetrics object with complete metrics from all tables
        """
        logger.info(f"Starting COMPLETE metrics extraction using ALL tuples from ALL tables... (backend: {backend_prefix})")
        
        # Extract complete hardware configuration using ALL TARGET_INFO_SYSTEM_ENV tuples
        hardware_config = self.extract_complete_hardware_configuration()
        
        # Extract complete ENUM mappings using ALL ENUM tables
        enum_mappings = self.extract_complete_enum_mappings()
        
        # Extract complete performance events using ALL event tables
        performance_events = self.extract_complete_performance_events()
        
        # Extract traditional metrics (enhanced with complete data)
        mpi_metrics = self.extract_mpi_communication_metrics(backend_prefix=backend_prefix)
        thread_metrics = self.extract_thread_performance_metrics()
        memory_metrics = self.extract_memory_locality_metrics()
        
        # Extract enhanced metrics
        gpu_metrics = self.extract_gpu_metrics()
        openmp_metrics = self.extract_openmp_metrics()
        enhanced_memory_metrics = self.extract_enhanced_memory_metrics()
        
        # Get total runtime
        total_runtime = self._get_total_runtime()
        
        # Use complete performance events data for enhanced metrics
        osrt_events = performance_events.get('osrt_api_events', {})
        cuda_kernels = performance_events.get('cuda_kernel_events', {})
        cuda_memcpy = performance_events.get('cuda_memcpy_events', {})
        sched_events = performance_events.get('sched_events', {})
        composite_events = performance_events.get('composite_events', {})
        
        # Enhanced MPI metrics using complete OSRT_API analysis
        complete_mpi_time = 0.0
        complete_mpi_calls = 0
        if 'event_classes' in osrt_events:
            # Use eventClass 27 for MPI events (OSRuntime)
            if 27 in osrt_events['event_classes']:
                mpi_class = osrt_events['event_classes'][27]
                complete_mpi_time = mpi_class.get('total_duration', 0.0)
                complete_mpi_calls = mpi_class.get('event_count', 0)
        
        # Enhanced GPU metrics using complete CUDA analysis
        complete_gpu_kernel_time = cuda_kernels.get('total_kernel_time', 0.0)
        complete_gpu_kernel_count = cuda_kernels.get('kernel_count', 0)
        complete_gpu_memcpy_time = cuda_memcpy.get('total_time', 0.0)
        complete_gpu_memcpy_count = cuda_memcpy.get('total_transfers', 0)
        
        # Enhanced thread metrics using complete scheduling analysis
        complete_context_switches = sched_events.get('total_events', 0)
        complete_thread_count = sched_events.get('unique_threads', 0)
        
        # Enhanced memory metrics using complete composite events
        complete_total_cycles = composite_events.get('total_cycles', 0)
        complete_thread_states = composite_events.get('thread_states', {})
        
        # Calculate enhanced CPU utilization from complete thread states
        enhanced_cpu_utilization = 0.0
        if complete_thread_states:
            # Thread state 1 typically represents running state
            running_cycles = complete_thread_states.get(1, {}).get('total_cycles', 0)
            if complete_total_cycles > 0:
                enhanced_cpu_utilization = running_cycles / complete_total_cycles
                logger.debug(f"Enhanced CPU utilization from thread states: {enhanced_cpu_utilization:.3f}")
        
        # Get base CPU utilization from thread metrics (stall-based calculation)
        base_cpu_utilization = thread_metrics.get('cpu_utilization', 0.0)
        
        # Prefer stall-based CPU util when both exist (thread-state can read "always running" under Nsight).
        if base_cpu_utilization > 0 and enhanced_cpu_utilization > 0:
            diff = abs(base_cpu_utilization - enhanced_cpu_utilization)
            if diff > 0.1:
                logger.debug(
                    "CPU util: stall-based=%.3f vs thread-state=%.3f — using stall-based",
                    base_cpu_utilization,
                    enhanced_cpu_utilization,
                )
            # Prefer stall-based calculation as it's more accurate
            final_cpu_utilization = base_cpu_utilization
        elif base_cpu_utilization > 0:
            final_cpu_utilization = base_cpu_utilization
        elif enhanced_cpu_utilization > 0:
            final_cpu_utilization = enhanced_cpu_utilization
        else:
            final_cpu_utilization = 0.0  # Unknown, not 1.0
        
        # MPI comm time: prefer mpiP (PMPI-level) when available; else Nsight OSRT
        if mpi_metrics.get('backend_detected') == 'mpiP' and mpi_metrics.get('total_comm_time', 0) > 0:
            final_mpi_comm_time = mpi_metrics['total_comm_time']
        else:
            final_mpi_comm_time = max(complete_mpi_time, mpi_metrics.get('total_comm_time', 0.0))
        mpi_comm_frac = 0.0
        if mpi_metrics.get("backend_detected") == "mpiP":
            mpi_comm_frac = float(mpi_metrics.get("mpip_max_mpi_wall_fraction", 0.0) or 0.0)
        if mpi_comm_frac <= 0.0 and final_mpi_comm_time > 0 and total_runtime > 0:
            mpi_comm_frac = min(1.0, final_mpi_comm_time / total_runtime)
        # ε (OpenMP): single definition from trace scheduling + CPU signals
        if openmp_metrics.get("openmp_instrumented"):
            openmp_eff = float(openmp_metrics.get("openmp_work_efficiency", 0.0) or 0.0)
            openmp_instrumented = True
            logger.info(
                f"OpenMP work efficiency (instrumented): {openmp_eff:.3f} "
                f"work={openmp_metrics.get('openmp_work_time', 0):.3f}s "
                f"sync={openmp_metrics.get('openmp_sync_time', 0):.3f}s"
            )
        else:
            openmp_eff = self._openmp_work_efficiency_from_trace(
                thread_metrics,
                total_runtime,
                sched_events,
                final_cpu_utilization,
                mpi_comm_fraction=mpi_comm_frac,
            )
            openmp_instrumented = False
            logger.info(
                f"OpenMP work efficiency (trace-based scheduling): {openmp_eff:.3f} "
                f"(stall={thread_metrics.get('thread_stall_time', 0) or 0:.3f}s, rt={total_runtime:.3f}s, "
                f"cpu_util={final_cpu_utilization:.3f}, sched_events={sched_events.get('total_events', 0)})"
            )
        # Combine all metrics with complete dataset insights
        combined_metrics = ExtractedMetrics(
            # Traditional metrics (enhanced with complete data)
            mpi_comm_time=final_mpi_comm_time,
            mpip_app_time=mpi_metrics.get('max_rank_app_time', 0.0),
            mpip_max_mpi_wall_fraction=float(mpi_metrics.get('mpip_max_mpi_wall_fraction', 0.0) or 0.0),
            total_runtime=total_runtime,
            cpu_utilization=final_cpu_utilization,
            memory_local_accesses=memory_metrics.get('memory_local_accesses', 0),
            memory_total_accesses=memory_metrics.get('memory_total_accesses', 0),
            numa_efficiency=memory_metrics.get('numa_efficiency', 0.0),
            locality_source=(
                "numa_meminfo"
                if memory_metrics.get("numa_source") == "numa_meminfo"
                else (
                    "dram_local_remote"
                    if (memory_metrics.get("memory_total_accesses", 0) or 0) > 0
                    else (
                        "l3_estimate"
                        if (memory_metrics.get("numa_efficiency", 0.0) or 0.0) > 0
                        else "missing"
                    )
                )
            ),
            thread_stall_time=thread_metrics.get('thread_stall_time', 0.0),
            load_imbalance=thread_metrics.get('load_imbalance', 0.0),
            cache_miss_rate=memory_metrics.get('cache_miss_rate', 0.0),
            mpi_call_count=max(complete_mpi_calls, mpi_metrics.get('mpi_call_count', 0)),
            mpi_message_size_total=mpi_metrics.get('total_data_transferred', 0),
            thread_context_switches=max(complete_context_switches, thread_metrics.get('total_sched_events', 0)),
            memory_bandwidth=memory_metrics.get('memory_bandwidth', 0.0),
            
            # Enhanced GPU metrics using complete CUDA analysis
            gpu_kernel_time=max(complete_gpu_kernel_time, gpu_metrics.get('gpu_kernel_time', 0.0)),
            gpu_memcpy_time=max(complete_gpu_memcpy_time, gpu_metrics.get('gpu_memcpy_time', 0.0)),
            gpu_sync_time=gpu_metrics.get('gpu_sync_time', 0.0),
            gpu_busy_time=max(
                float(gpu_metrics.get('gpu_busy_time', 0.0) or 0.0),
                complete_gpu_kernel_time + complete_gpu_memcpy_time + float(gpu_metrics.get('gpu_sync_time', 0.0) or 0.0),
            ),
            gpu_utilization=gpu_metrics.get('gpu_utilization', 0.0),
            gpu_memory_bandwidth=gpu_metrics.get('gpu_memory_bandwidth', 0.0),
            gpu_kernel_count=max(complete_gpu_kernel_count, gpu_metrics.get('gpu_kernel_count', 0)),
            gpu_memcpy_count=max(complete_gpu_memcpy_count, gpu_metrics.get('gpu_memcpy_count', 0)),
            gpu_active_span=float(gpu_metrics.get('gpu_active_span', 0.0) or 0.0),
            cupti_process_count=int(gpu_metrics.get('cupti_process_count', 0) or 0),
            
            # OpenMP ε from trace (see _openmp_work_efficiency_from_trace)
            openmp_work_time=openmp_metrics.get('openmp_work_time', 0.0),
            openmp_sync_time=openmp_metrics.get('openmp_sync_time', 0.0),
            openmp_task_count=openmp_metrics.get('openmp_task_count', 0),
            openmp_work_efficiency=openmp_eff,
            openmp_instrumented=openmp_instrumented,
            
            # Enhanced memory metrics
            memory_transfer_rate=enhanced_memory_metrics.get('memory_transfer_rate', 0.0),
            memory_bottleneck_severity=enhanced_memory_metrics.get('memory_bottleneck_severity', 0.0)
        )
        
        # Store complete analysis data for advanced reporting
        combined_metrics.hardware_configuration = hardware_config
        combined_metrics.enum_mappings = enum_mappings
        combined_metrics.performance_events = performance_events
        
        # Cap MPI comm time at 60% only for OSRT/syscall estimate (mpiP is authoritative; don't cap it)
        if combined_metrics.total_runtime > 0 and mpi_metrics.get('backend_detected') != 'mpiP':
            cap = 0.60 * combined_metrics.total_runtime
            if combined_metrics.mpi_comm_time > cap:
                combined_metrics.mpi_comm_time = cap
                logger.debug(f"Capped mpi_comm_time at 60% of runtime (syscall estimate) for α accuracy")

        # δ (GPU): refine with application wall, mpiP fraction, multi-rank CUPTI coverage
        from autotuner.core.profiling_refinement import refine_gpu_utilization

        app_wall = self._get_wall_time_from_run_artifacts()
        trace_wall = max(0.0, float(combined_metrics.total_runtime or 0.0))
        mpi_frac = float(getattr(combined_metrics, "mpip_max_mpi_wall_fraction", 0.0) or 0.0)
        if mpi_frac <= 0.001 and trace_wall > 0:
            mpi_frac = min(
                1.0,
                max(0.0, float(combined_metrics.mpi_comm_time or 0.0)) / trace_wall,
            )
        hw = combined_metrics.hardware_configuration or {}
        mpi_ranks = int(hw.get("mpi_ranks") or hw.get("total_mpi_ranks") or 1)
        gu, gu_src = refine_gpu_utilization(
            float(combined_metrics.gpu_busy_time or 0.0),
            application_runtime=app_wall,
            trace_runtime=trace_wall,
            mpi_comm_time=float(combined_metrics.mpi_comm_time or 0.0),
            mpip_wall_fraction=mpi_frac,
            total_mpi_ranks=max(1, mpi_ranks),
            cupti_process_count=int(combined_metrics.cupti_process_count or 0),
            gpu_active_span=float(combined_metrics.gpu_active_span or 0.0),
            cpu_utilization=float(combined_metrics.cpu_utilization or 0.0),
            profiling_phase_aligned=app_wall > 0 and abs(app_wall - trace_wall) / max(app_wall, 1e-6) < 0.35,
        )
        if gu > 0:
            prev = float(combined_metrics.gpu_utilization or 0.0)
            combined_metrics.gpu_utilization = gu
            combined_metrics.gpu_utilization_source = gu_src
            if abs(gu - prev) > 0.02:
                logger.info(
                    "GPU utilization refined: %.4f → %.4f (%s)",
                    prev,
                    gu,
                    gu_src,
                )
        
        # Extract distribution analysis for bottleneck detection
        try:
            distribution_analysis = self.extract_distribution_analysis()
            combined_metrics.distribution_stats = {
                'mpi': distribution_analysis.get('mpi_distribution', {}).get('distribution_stats', {}),
                'thread': distribution_analysis.get('thread_distribution', {}).get('distribution_stats', {}),
                'gpu': distribution_analysis.get('gpu_distribution', {}).get('distribution_stats', {})
            }
            combined_metrics.per_rank_breakdown = distribution_analysis.get('mpi_distribution', {})
            combined_metrics.per_thread_breakdown = distribution_analysis.get('thread_distribution', {})
            combined_metrics.per_stream_breakdown = distribution_analysis.get('gpu_distribution', {})
            combined_metrics.outliers = distribution_analysis.get('outliers', {})
            combined_metrics.bottleneck_details = distribution_analysis.get('bottleneck_summary', {})
            
            logger.debug("Distribution analysis completed")
            if combined_metrics.bottleneck_details:
                bottleneck_summary = combined_metrics.bottleneck_details
                logger.debug(f"  Bottleneck severity: {bottleneck_summary.get('overall_severity', 'none')}")
                logger.debug(f"  Communication bottlenecks: {bottleneck_summary.get('communication_bottlenecks', {}).get('count', 0)} ranks")
                logger.debug(f"  Threading bottlenecks: {bottleneck_summary.get('threading_bottlenecks', {}).get('count', 0)} threads")
                logger.debug(f"  GPU bottlenecks: {bottleneck_summary.get('gpu_bottlenecks', {}).get('count', 0)} streams")
        except Exception as e:
            logger.warning(f"Distribution analysis failed (non-critical): {e}")
            # Set defaults if distribution analysis fails
            combined_metrics.distribution_stats = None
            combined_metrics.per_rank_breakdown = None
            combined_metrics.per_thread_breakdown = None
            combined_metrics.per_stream_breakdown = None
            combined_metrics.outliers = None
            combined_metrics.bottleneck_details = None
        
        logger.debug(f"Metrics extraction completed: {hardware_config['cpu_cores']} cores, {osrt_events.get('total_events', 0)} OSRT events")
        
        return combined_metrics
    
    def export_metrics_report(self, output_path: Path, metrics: ExtractedMetrics) -> None:
        """
        Export extracted metrics to a detailed report
        
        Args:
            output_path: Directory to save the report
            metrics: Extracted metrics object
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create detailed metrics report
        report_data = {
            'extraction_timestamp': time.time(),
            'database_path': str(self.database_path),
            'metrics': {
                'mpi_communication': {
                    'comm_time': metrics.mpi_comm_time,
                    'comm_fraction': min(metrics.mpi_comm_time / metrics.total_runtime, 1.0) if metrics.total_runtime > 0 else 0,
                    'call_count': metrics.mpi_call_count,
                    'total_message_size': metrics.mpi_message_size_total
                },
                'thread_performance': {
                    'cpu_utilization': metrics.cpu_utilization,
                    'stall_time': metrics.thread_stall_time,
                    'stall_fraction': metrics.thread_stall_time / metrics.total_runtime if metrics.total_runtime > 0 else 0,
                    'context_switches': metrics.thread_context_switches
                },
                'memory_locality': {
                    'local_accesses': metrics.memory_local_accesses,
                    'total_accesses': metrics.memory_total_accesses,
                    'numa_efficiency': metrics.numa_efficiency,
                    'cache_miss_rate': metrics.cache_miss_rate
                },
                'overall': {
                    'total_runtime': metrics.total_runtime,
                    'load_imbalance': metrics.load_imbalance,
                    'memory_bandwidth': metrics.memory_bandwidth
                },
                'gpu_performance': {
                    'gpu_utilization': metrics.gpu_utilization,
                    'gpu_kernel_time': metrics.gpu_kernel_time,
                    'gpu_memcpy_time': metrics.gpu_memcpy_time,
                    'gpu_sync_time': metrics.gpu_sync_time,
                    'gpu_memory_bandwidth': metrics.gpu_memory_bandwidth,
                    'gpu_kernel_count': metrics.gpu_kernel_count,
                    'gpu_memcpy_count': metrics.gpu_memcpy_count
                },
                'openmp_performance': {
                    'openmp_work_time': metrics.openmp_work_time,
                    'openmp_sync_time': metrics.openmp_sync_time,
                    'openmp_task_count': metrics.openmp_task_count,
                    'openmp_work_efficiency': metrics.openmp_work_efficiency
                },
                'enhanced_memory': {
                    'memory_transfer_rate': metrics.memory_transfer_rate,
                    'memory_bottleneck_severity': metrics.memory_bottleneck_severity
                }
            }
        }
        
        # Save as JSON
        json_file = output_path / "extracted_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save as CSV
        csv_data = []
        for category, category_metrics in report_data['metrics'].items():
            for metric_name, metric_value in category_metrics.items():
                csv_data.append({
                    'category': category,
                    'metric': metric_name,
                    'value': metric_value
                })
        
        df = pd.DataFrame(csv_data)
        csv_file = output_path / "extracted_metrics.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Metrics report exported to {output_path}")
        logger.info(f"  - JSON: {json_file}")
        logger.info(f"  - CSV: {csv_file}")
    
    def extract_application_metrics(self, output_file: Path) -> Dict[str, float]:
        """
        Extract application-level metrics from output files (e.g., GFLOPS, time-to-solution)
        
        This method parses application output files to extract high-level performance metrics
        that complement the low-level profiling metrics from Nsight Systems.
        
        Args:
            output_file: Path to application output file (stdout/stderr)
            
        Returns:
            Dictionary with application metrics:
            - 'gflops': Floating-point operations per second (in GFLOPS)
            - 'time_to_solution': Total execution time in seconds
            - 'throughput': Problem size / time_to_solution
            - 'efficiency': Actual performance / theoretical peak
        """
        app_metrics = {
            'gflops': 0.0,
            'time_to_solution': 0.0,
            'throughput': 0.0,
            'efficiency': 0.0
        }
        
        if not output_file.exists():
            logger.warning(f"Application output file not found: {output_file}")
            return app_metrics
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Pattern matching for common performance metrics
            import re
            
            # Extract GFLOPS (various formats: "GFLOPS: 123.45", "123.45 GFLOPS", "Performance: 123.45 GFlop/s")
            gflops_patterns = [
                r'GFLOPS[:\s=]+([\d.]+)',
                r'([\d.]+)\s*GFLOPS?',
                r'([\d.]+)\s*GFlop/s',
                r'Performance[:\s=]+([\d.]+)\s*G',
                r'FLOPS[:\s=]+([\d.]+)\s*G'
            ]
            
            for pattern in gflops_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    app_metrics['gflops'] = float(match.group(1))
                    break
            
            # Extract time-to-solution (various formats: "Time: 123.45s", "Elapsed: 123.45", "Total time: 123.45 seconds")
            time_patterns = [
                r'Time[:\s=]+([\d.]+)\s*s',
                r'Elapsed[:\s=]+([\d.]+)',
                r'Total\s+time[:\s=]+([\d.]+)',
                r'Runtime[:\s=]+([\d.]+)',
                r'Execution\s+time[:\s=]+([\d.]+)',
                r'Wall\s+time[:\s=]+([\d.]+)'
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    app_metrics['time_to_solution'] = float(match.group(1))
                    break
            
            # Extract problem size (for throughput calculation)
            size_patterns = [
                r'Size[:\s=]+(\d+)',
                r'Problem\s+size[:\s=]+(\d+)',
                r'Grid\s+size[:\s=]+(\d+)',
                r'N[:\s=]+(\d+)'
            ]
            
            problem_size = 0
            for pattern in size_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    problem_size = int(match.group(1))
                    break
            
            # Calculate throughput if we have problem size and time
            if problem_size > 0 and app_metrics['time_to_solution'] > 0:
                app_metrics['throughput'] = problem_size / app_metrics['time_to_solution']
            
            # Estimate efficiency if we have GFLOPS (assuming theoretical peak)
            # For CPU: rough estimate based on core count and frequency
            # This is a placeholder - actual peak should be calculated from hardware specs
            if app_metrics['gflops'] > 0:
                # Conservative estimate: assume 100 GFLOPS per core as theoretical peak
                # This should be replaced with actual hardware specifications
                estimated_peak = 100.0  # Placeholder
                app_metrics['efficiency'] = min(app_metrics['gflops'] / estimated_peak, 1.0)
            
            logger.info(f"Extracted application metrics from {output_file}:")
            logger.info(f"  GFLOPS: {app_metrics['gflops']:.2f}")
            logger.info(f"  Time-to-solution: {app_metrics['time_to_solution']:.2f}s")
            logger.info(f"  Throughput: {app_metrics['throughput']:.2f}")
            logger.info(f"  Efficiency: {app_metrics['efficiency']:.2%}")
            
        except Exception as e:
            logger.error(f"Error extracting application metrics from {output_file}: {e}")
        
        return app_metrics
    
    def extract_distribution_analysis(self) -> Dict[str, Any]:
        """
        Extract distribution analysis for bottleneck detection
        
        Returns:
            Dictionary containing distribution statistics, per-rank/thread breakdowns,
            outliers, and bottleneck details
        """
        logger.info("Extracting distribution analysis for bottleneck detection...")
        
        distribution_data = {
            'mpi_distribution': self._extract_mpi_distribution(),
            'thread_distribution': self._extract_thread_distribution(),
            'gpu_distribution': self._extract_gpu_distribution(),
            'numa_distribution': self._extract_numa_distribution(),
            'outliers': self._detect_outliers(),
            'bottleneck_summary': {}
        }
        
        # Generate bottleneck summary
        distribution_data['bottleneck_summary'] = self._generate_bottleneck_summary(distribution_data)
        
        return distribution_data
    
    def _extract_mpi_distribution(self) -> Dict[str, Any]:
        """Extract per-rank MPI communication distribution"""
        if not self.connection:
            self.connect_database()
        
        executor = self.connection.cursor()
        mpi_dist = {
            'per_rank_comm': [],
            'distribution_stats': {},
            'bottleneck_ranks': []
        }
        
        try:
            if 'OSRT_API' in self.get_available_tables():
                osrt_cols = set(self._get_table_columns('OSRT_API'))
                has_stringids = 'StringIds' in self.get_available_tables()
                
                if {'start', 'end', 'globalTid', 'nameId'}.issubset(osrt_cols) and has_stringids:
                    # Per-rank MPI communication breakdown
                    per_rank_query = """
                    SELECT 
                        o.globalTid,
                        COUNT(*) as call_count,
                        SUM((o.end - o.start)/1e9) as total_time,
                        AVG((o.end - o.start)/1e9) as avg_time,
                        MIN((o.end - o.start)/1e9) as min_time,
                        MAX((o.end - o.start)/1e9) as max_time
                    FROM OSRT_API o
                    LEFT JOIN StringIds s ON o.nameId = s.id
                    WHERE s.value LIKE '%MPI%' OR s.value LIKE '%mpi%'
                    GROUP BY o.globalTid
                    ORDER BY total_time DESC
                    """
                    executor.execute(per_rank_query)
                    rows = executor.fetchall()
                    
                    per_rank_data = []
                    times = []
                    for row in rows:
                        rank_data = {
                            'rank_id': row['globalTid'],
                            'call_count': row['call_count'] or 0,
                            'total_time': row['total_time'] or 0.0,
                            'avg_time': row['avg_time'] or 0.0,
                            'min_time': row['min_time'] or 0.0,
                            'max_time': row['max_time'] or 0.0
                        }
                        per_rank_data.append(rank_data)
                        times.append(row['total_time'] or 0.0)
                    
                    mpi_dist['per_rank_comm'] = per_rank_data
                    
                    # Calculate distribution statistics
                    if times:
                        times_array = np.array(times)
                        mpi_dist['distribution_stats'] = {
                            'mean': float(np.mean(times_array)),
                            'median': float(np.median(times_array)),
                            'std_dev': float(np.std(times_array)),
                            'min': float(np.min(times_array)),
                            'max': float(np.max(times_array)),
                            'p50': float(np.percentile(times_array, 50)),
                            'p90': float(np.percentile(times_array, 90)),
                            'p95': float(np.percentile(times_array, 95)),
                            'p99': float(np.percentile(times_array, 99)),
                            'skewness': float(self._calculate_skewness(times_array)),
                            'cv': float(np.std(times_array) / np.mean(times_array)) if np.mean(times_array) > 0 else 0.0
                        }
                        
                        # Identify bottleneck ranks (ranks with comm time > mean + 2*std)
                        mean = mpi_dist['distribution_stats']['mean']
                        std = mpi_dist['distribution_stats']['std_dev']
                        threshold = mean + 2 * std
                        
                        for rank_data in per_rank_data:
                            if rank_data['total_time'] > threshold:
                                mpi_dist['bottleneck_ranks'].append({
                                    'rank_id': rank_data['rank_id'],
                                    'total_time': rank_data['total_time'],
                                    'deviation': rank_data['total_time'] - mean,
                                    'severity': 'high' if rank_data['total_time'] > mean + 3*std else 'medium'
                                })
        except Exception as e:
            logger.warning(f"Error extracting MPI distribution: {e}")
        
        return mpi_dist
    
    def _extract_thread_distribution(self) -> Dict[str, Any]:
        """Extract per-thread performance distribution"""
        if not self.connection:
            self.connect_database()
        
        executor = self.connection.cursor()
        thread_dist = {
            'per_thread_util': [],
            'distribution_stats': {},
            'bottleneck_threads': []
        }
        
        try:
            if 'SCHED_EVENTS' in self.get_available_tables():
                sched_cols = set(self._get_table_columns('SCHED_EVENTS'))
                
                if {'start', 'globalTid', 'isSchedIn'}.issubset(sched_cols):
                    # Per-thread utilization breakdown
                    per_thread_query = """
                    SELECT 
                        globalTid,
                        COUNT(*) as total_events,
                        SUM(isSchedIn) as sched_in_events,
                        COUNT(*) - SUM(isSchedIn) as sched_out_events
                    FROM SCHED_EVENTS
                    GROUP BY globalTid
                    """
                    executor.execute(per_thread_query)
                    rows = executor.fetchall()
                    
                    per_thread_data = []
                    utilizations = []
                    for row in rows:
                        total = row['total_events'] or 1
                        sched_in = row['sched_in_events'] or 0
                        utilization = sched_in / total if total > 0 else 0.0
                        
                        thread_data = {
                            'thread_id': row['globalTid'],
                            'total_events': total,
                            'sched_in_events': sched_in,
                            'sched_out_events': row['sched_out_events'] or 0,
                            'utilization': utilization
                        }
                        per_thread_data.append(thread_data)
                        utilizations.append(utilization)
                    
                    thread_dist['per_thread_util'] = per_thread_data
                    
                    # Calculate distribution statistics
                    if utilizations:
                        util_array = np.array(utilizations)
                        thread_dist['distribution_stats'] = {
                            'mean': float(np.mean(util_array)),
                            'median': float(np.median(util_array)),
                            'std_dev': float(np.std(util_array)),
                            'min': float(np.min(util_array)),
                            'max': float(np.max(util_array)),
                            'p50': float(np.percentile(util_array, 50)),
                            'p90': float(np.percentile(util_array, 90)),
                            'p95': float(np.percentile(util_array, 95)),
                            'p99': float(np.percentile(util_array, 99)),
                            'skewness': float(self._calculate_skewness(util_array)),
                            'cv': float(np.std(util_array) / np.mean(util_array)) if np.mean(util_array) > 0 else 0.0
                        }
                        
                        # Identify bottleneck threads (low utilization)
                        mean = thread_dist['distribution_stats']['mean']
                        std = thread_dist['distribution_stats']['std_dev']
                        threshold = mean - 2 * std
                        
                        for thread_data in per_thread_data:
                            if thread_data['utilization'] < threshold:
                                thread_dist['bottleneck_threads'].append({
                                    'thread_id': thread_data['thread_id'],
                                    'utilization': thread_data['utilization'],
                                    'deviation': thread_data['utilization'] - mean,
                                    'severity': 'high' if thread_data['utilization'] < mean - 3*std else 'medium'
                                })
        except Exception as e:
            logger.warning(f"Error extracting thread distribution: {e}")
        
        return thread_dist
    
    def _extract_gpu_distribution(self) -> Dict[str, Any]:
        """Extract per-stream GPU utilization distribution"""
        if not self.connection:
            self.connect_database()
        
        executor = self.connection.cursor()
        gpu_dist = {
            'per_stream_util': [],
            'distribution_stats': {},
            'bottleneck_streams': []
        }
        
        try:
            for kt in self._cuda_kernel_activity_tables():
                kernel_cols = set(self._get_table_columns(kt))
                pair = self._kernel_time_column_pair(kt)
                if not pair or "streamId" not in kernel_cols:
                    continue
                s, e = pair
                per_stream_query = f"""
                SELECT 
                    streamId,
                    COUNT(*) as kernel_count,
                    SUM(({e} - {s})/1e9) as total_time,
                    AVG(({e} - {s})/1e9) as avg_time,
                    MIN(({e} - {s})/1e9) as min_time,
                    MAX(({e} - {s})/1e9) as max_time
                FROM {kt}
                WHERE {e} > {s}
                GROUP BY streamId
                ORDER BY total_time DESC
                """
                try:
                    executor.execute(per_stream_query)
                    rows = executor.fetchall()
                except Exception:
                    continue
                if not rows:
                    continue

                per_stream_data = []
                times = []
                for row in rows:
                    stream_data = {
                        'stream_id': row['streamId'],
                        'kernel_count': row['kernel_count'] or 0,
                        'total_time': row['total_time'] or 0.0,
                        'avg_time': row['avg_time'] or 0.0,
                        'min_time': row['min_time'] or 0.0,
                        'max_time': row['max_time'] or 0.0
                    }
                    per_stream_data.append(stream_data)
                    times.append(row['total_time'] or 0.0)

                gpu_dist['per_stream_util'] = per_stream_data

                if times:
                    times_array = np.array(times)
                    gpu_dist['distribution_stats'] = {
                        'mean': float(np.mean(times_array)),
                        'median': float(np.median(times_array)),
                        'std_dev': float(np.std(times_array)),
                        'min': float(np.min(times_array)),
                        'max': float(np.max(times_array)),
                        'p50': float(np.percentile(times_array, 50)),
                        'p90': float(np.percentile(times_array, 90)),
                        'p95': float(np.percentile(times_array, 95)),
                        'p99': float(np.percentile(times_array, 99)),
                        'skewness': float(self._calculate_skewness(times_array)),
                        'cv': float(np.std(times_array) / np.mean(times_array)) if np.mean(times_array) > 0 else 0.0
                    }

                    mean = gpu_dist['distribution_stats']['mean']
                    std = gpu_dist['distribution_stats']['std_dev']
                    threshold = mean + 2 * std

                    for stream_data in per_stream_data:
                        if stream_data['total_time'] > threshold:
                            gpu_dist['bottleneck_streams'].append({
                                'stream_id': stream_data['stream_id'],
                                'total_time': stream_data['total_time'],
                                'deviation': stream_data['total_time'] - mean,
                                'severity': 'high' if stream_data['total_time'] > mean + 3*std else 'medium'
                            })
                break
        except Exception as e:
            logger.warning(f"Error extracting GPU distribution: {e}")
        
        return gpu_dist
    
    def _extract_numa_distribution(self) -> Dict[str, Any]:
        """Extract NUMA access distribution (if available from LIKWID)"""
        numa_dist = {
            'per_socket_access': [],
            'distribution_stats': {},
            'bottleneck_sockets': []
        }
        
        # This would be enhanced with LIKWID data if available
        # For now, return empty structure
        return numa_dist
    
    def _detect_outliers(self) -> Dict[str, List[Any]]:
        """Detect outliers across all metrics"""
        outliers = {
            'mpi_outliers': [],
            'thread_outliers': [],
            'gpu_outliers': []
        }
        
        # Outlier detection is done in individual distribution methods
        # This method aggregates them
        return outliers
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness coefficient"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return float(skew)
    
    def _generate_bottleneck_summary(self, distribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all bottlenecks"""
        summary = {
            'communication_bottlenecks': {
                'count': len(distribution_data.get('mpi_distribution', {}).get('bottleneck_ranks', [])),
                'severity': 'none',
                'affected_ranks': []
            },
            'threading_bottlenecks': {
                'count': len(distribution_data.get('thread_distribution', {}).get('bottleneck_threads', [])),
                'severity': 'none',
                'affected_threads': []
            },
            'gpu_bottlenecks': {
                'count': len(distribution_data.get('gpu_distribution', {}).get('bottleneck_streams', [])),
                'severity': 'none',
                'affected_streams': []
            },
            'overall_severity': 'none',
            'recommendations': []
        }
        
        # Determine severity
        mpi_bottlenecks = distribution_data.get('mpi_distribution', {}).get('bottleneck_ranks', [])
        thread_bottlenecks = distribution_data.get('thread_distribution', {}).get('bottleneck_threads', [])
        gpu_bottlenecks = distribution_data.get('gpu_distribution', {}).get('bottleneck_streams', [])
        
        if mpi_bottlenecks:
            high_severity = sum(1 for b in mpi_bottlenecks if b.get('severity') == 'high')
            summary['communication_bottlenecks']['severity'] = 'high' if high_severity > 0 else 'medium'
            summary['communication_bottlenecks']['affected_ranks'] = [b['rank_id'] for b in mpi_bottlenecks]
        
        if thread_bottlenecks:
            high_severity = sum(1 for b in thread_bottlenecks if b.get('severity') == 'high')
            summary['threading_bottlenecks']['severity'] = 'high' if high_severity > 0 else 'medium'
            summary['threading_bottlenecks']['affected_threads'] = [b['thread_id'] for b in thread_bottlenecks]
        
        if gpu_bottlenecks:
            high_severity = sum(1 for b in gpu_bottlenecks if b.get('severity') == 'high')
            summary['gpu_bottlenecks']['severity'] = 'high' if high_severity > 0 else 'medium'
            summary['gpu_bottlenecks']['affected_streams'] = [b['stream_id'] for b in gpu_bottlenecks]
        
        # Overall severity
        severities = [
            summary['communication_bottlenecks']['severity'],
            summary['threading_bottlenecks']['severity'],
            summary['gpu_bottlenecks']['severity']
        ]
        if 'high' in severities:
            summary['overall_severity'] = 'high'
        elif 'medium' in severities:
            summary['overall_severity'] = 'medium'
        
        # Generate recommendations
        if summary['communication_bottlenecks']['count'] > 0:
            summary['recommendations'].append(
                f"Redistribute MPI work: {summary['communication_bottlenecks']['count']} ranks show high communication overhead"
            )
        if summary['threading_bottlenecks']['count'] > 0:
            summary['recommendations'].append(
                f"Optimize thread scheduling: {summary['threading_bottlenecks']['count']} threads show low utilization"
            )
        if summary['gpu_bottlenecks']['count'] > 0:
            summary['recommendations'].append(
                f"Balance GPU work: {summary['gpu_bottlenecks']['count']} streams show high utilization"
            )
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m autotuner.core.metrics_extractor <database_path> [output_dir]")
        sys.exit(1)
    
    db_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("extracted_metrics")
    
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    try:
        with NsightMetricsExtractor(db_path) as extractor:
            # Extract all metrics
            metrics = extractor.extract_all_metrics()
            
            # Print summary
            print(f"\nExtracted Metrics Summary:")
            print(f"MPI Comm Time: {metrics.mpi_comm_time:.2f}s")
            print(f"Total Runtime: {metrics.total_runtime:.2f}s")
            print(f"CPU Utilization: {metrics.cpu_utilization:.3f}")
            print(f"NUMA Efficiency: {metrics.numa_efficiency:.3f}")
            print(f"MPI Call Count: {metrics.mpi_call_count}")
            
            # Export report
            extractor.export_metrics_report(output_dir, metrics)
            print(f"\n✅ Metrics report exported to: {output_dir}")
            
    except Exception as e:
        print(f"Error during metrics extraction: {e}")
        sys.exit(1)
