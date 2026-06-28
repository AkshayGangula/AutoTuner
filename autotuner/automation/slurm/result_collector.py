#!/usr/bin/env python3
"""
SLURM Result Collection and Management

Handles collection, organization, and cleanup of SLURM job results and profiling data.
"""

import json
import os
import shutil
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from autotuner.utils.path_utils import (
    get_logical_cwd,
    experiment_artifact_dir_paths,
    experiment_path_roots,
    slurm_log_paths_from_sacct,
    slurm_log_search_dirs,
    log_sacct_job_diagnostic,
)

logger = logging.getLogger(__name__)


class SLURMResultCollector:
    """Collects and manages SLURM job results"""

    def __init__(self, work_directory: Path, system_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the result collector

        Args:
            work_directory: Working directory for results
            system_config: HPC system dict (``artifact_dir``, ``slurm_log_dir``, …)
        """
        p = Path(work_directory)
        base = get_logical_cwd()
        wd = (base / p) if not p.is_absolute() else p
        try:
            self.work_directory = wd.expanduser().resolve()
        except OSError:
            self.work_directory = wd.expanduser()
        self.system_config = system_config or {}
        self.results_dir = self.work_directory / "results"
        self.profiling_dir = self.work_directory / "profiling"
        self.logs_dir = self.work_directory / "logs"
        
        # Create directories
        for directory in [self.results_dir, self.profiling_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def collect_results(self, job_id: str, job_history: Dict[str, Any]) -> Path:
        """
        Collect results from a completed job
        
        Args:
            job_id: SLURM job ID
            job_history: Job history dictionary
            
        Returns:
            Path to results directory
        """
        results_dir = self.results_dir / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy SLURM output files
        self._copy_slurm_files(job_id, results_dir)
        if not any(results_dir.glob("slurm-*")):
            log_sacct_job_diagnostic(
                job_id,
                prefix="No slurm-*.out/.err under results dir — ",
            )
        
        # Copy Run 1 output (multi-rank light application) if present
        self._copy_run1_output(job_id, results_dir)
        
        # Copy profiling data if available
        self._copy_profiling_data(job_id, results_dir)
        
        # LIKWID CSVs live under experiment_dir/likwid_output/<job_id> (same FS as profiling)
        self._copy_likwid_output(job_id, results_dir)
        self._copy_mpip_output(job_id, results_dir)
        
        # Copy job script
        self._copy_job_script(job_id, job_history, results_dir)
        
        # Create results summary
        self._create_results_summary(job_id, results_dir)
        
        logger.info(f"Collected results for job {job_id} to {results_dir}")
        return results_dir
    
    def _artifact_roots(self) -> List[Path]:
        """Experiment tree + optional scratch ``artifact_dir`` (profiling, LIKWID, run1, mpiP)."""
        try:
            return experiment_artifact_dir_paths(self.work_directory, self.system_config)
        except Exception:
            return [self.work_directory]

    def _slurm_search_roots(self) -> List[Path]:
        """Experiment dirs that may contain slurm-<job_id>.* (/home vs backing mount, resolve())."""
        roots: List[Path] = []
        try:
            roots.extend(experiment_path_roots(self.work_directory))
        except Exception:
            roots.append(self.work_directory)
        roots.extend(slurm_log_search_dirs(self.system_config))
        seen = set()
        out: List[Path] = []
        for r in roots:
            try:
                rp = Path(r).expanduser().resolve()
            except OSError:
                rp = Path(r).expanduser()
            key = str(rp)
            if key not in seen:
                seen.add(key)
                out.append(rp)
        return out

    def _copy_slurm_files(self, job_id: str, results_dir: Path) -> None:
        """Copy SLURM stdout/stderr into results/<job_id>/. Retries + alternate roots + sacct fallback."""
        slurm_files: List[Path] = []
        delays = (2, 5, 8)
        for attempt in range(len(delays) + 1):
            seen_paths: set[str] = set()
            for root in self._slurm_search_roots():
                for p in root.glob(f"slurm-{job_id}.*"):
                    if p.is_file():
                        key = str(p.resolve())
                        if key not in seen_paths:
                            seen_paths.add(key)
                            slurm_files.append(p)
                for ext in (".out", ".err"):
                    p = root / f"slurm-{job_id}{ext}"
                    if p.is_file():
                        key = str(p.resolve())
                        if key not in seen_paths:
                            seen_paths.add(key)
                            slurm_files.append(p)
            if slurm_files:
                break
            if attempt < len(delays):
                time.sleep(delays[attempt])

        if not slurm_files:
            sacct_list = slurm_log_paths_from_sacct(job_id)
            if sacct_list:
                logger.info(
                    "Found %d SLURM log path(s) via sacct for job %s (glob miss)",
                    len(sacct_list),
                    job_id,
                )
            for p in sacct_list:
                slurm_files.append(p)

        for slurm_file in slurm_files:
            try:
                shutil.copy2(slurm_file, results_dir / slurm_file.name)
                logger.debug(f"Copied SLURM file: {slurm_file.name}")
            except Exception as e:
                logger.warning(f"Failed to copy SLURM file {slurm_file}: {e}")

    def _copy_run1_output(self, job_id: str, results_dir: Path) -> None:
        """Copy Run 1 output file (run1_<job_id>.out) to results directory if present."""
        run1_name = f"run1_{job_id}.out"
        for root in self._artifact_roots():
            run1_src = root / run1_name
            if run1_src.is_file():
                try:
                    shutil.copy2(run1_src, results_dir / run1_name)
                    logger.debug(f"Copied Run 1 output: {run1_name} from {root}")
                except Exception as e:
                    logger.warning(f"Failed to copy Run 1 output {run1_src}: {e}")
                return
    
    def _copy_profiling_data(self, job_id: str, results_dir: Path) -> None:
        """Copy profiling data to results directory"""
        profiling_dir = None
        for root in self._artifact_roots():
            candidate = root / "profiling" / job_id
            if candidate.is_dir():
                profiling_dir = candidate
                break
        if profiling_dir is None:
            profiling_dir = self.profiling_dir / job_id
        if profiling_dir.exists():
            try:
                shutil.copytree(profiling_dir, results_dir / "profiling", dirs_exist_ok=True)
                logger.debug(f"Copied profiling data for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to copy profiling data for job {job_id}: {e}")
    
    def _copy_likwid_output(self, job_id: str, results_dir: Path) -> None:
        """Copy LIKWID counter output so results/<job_id>/likwid mirrors experiment layout for archiving."""
        likwid_src = None
        for root in self._artifact_roots():
            candidate = root / "likwid_output" / job_id
            if candidate.is_dir():
                likwid_src = candidate
                break
        if likwid_src is None:
            likwid_src = self.work_directory / "likwid_output" / job_id
        if not likwid_src.is_dir():
            return
        likwid_dst = results_dir / "likwid_output"
        try:
            shutil.copytree(likwid_src, likwid_dst, dirs_exist_ok=True)
            logger.debug(f"Copied LIKWID output for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to copy LIKWID output for job {job_id}: {e}")
    
    def _copy_mpip_output(self, job_id: str, results_dir: Path) -> None:
        """Copy mpiP report tree (Run 1a srun+LD_PRELOAD) for archiving."""
        mpip_src = None
        for root in self._artifact_roots():
            candidate = root / "mpiP" / job_id
            if candidate.is_dir():
                mpip_src = candidate
                break
        if mpip_src is None:
            mpip_src = self.work_directory / "mpiP" / job_id
        if not mpip_src.is_dir():
            return
        mpip_dst = results_dir / "mpiP"
        try:
            shutil.copytree(mpip_src, mpip_dst, dirs_exist_ok=True)
            logger.debug(f"Copied mpiP output for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to copy mpiP output for job {job_id}: {e}")
    
    def _copy_job_script(self, job_id: str, job_history: Dict[str, Any], results_dir: Path) -> None:
        """Copy job script to results directory"""
        if job_id in job_history:
            script_path = job_history[job_id].get('script_path')
            if script_path and Path(script_path).exists():
                try:
                    shutil.copy2(script_path, results_dir / "job_script.slurm")
                    logger.debug(f"Copied job script for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to copy job script for job {job_id}: {e}")
    
    def _create_results_summary(self, job_id: str, results_dir: Path) -> None:
        """Create a summary of job results"""
        summary = {
            'job_id': job_id,
            'collection_time': time.time(),
            'slurm_files': [],
            'profiling_data': [],
            'job_script': None
        }
        
        # List SLURM files
        for file_path in results_dir.glob("slurm-*"):
            summary['slurm_files'].append({
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'modified': file_path.stat().st_mtime
            })
        
        # List profiling data
        profiling_dir = results_dir / "profiling"
        if profiling_dir.exists():
            for file_path in profiling_dir.rglob("*"):
                if file_path.is_file():
                    summary['profiling_data'].append({
                        'name': str(file_path.relative_to(profiling_dir)),
                        'size': file_path.stat().st_size,
                        'type': file_path.suffix
                    })
        
        # Check for job script
        job_script = results_dir / "job_script.slurm"
        if job_script.exists():
            summary['job_script'] = {
                'name': job_script.name,
                'size': job_script.stat().st_size
            }
        
        # Save summary
        summary_file = results_dir / "results_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def cleanup_old_results(self, max_age_days: int = 30) -> None:
        """
        Clean up old result directories
        
        Args:
            max_age_days: Maximum age of results to keep (in days)
        """
        import time
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0
        
        for result_dir in self.results_dir.iterdir():
            if result_dir.is_dir():
                try:
                    # Check if directory is older than cutoff
                    if result_dir.stat().st_mtime < cutoff_time:
                        shutil.rmtree(result_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old results: {result_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {result_dir}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old result directories")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected results
        
        Returns:
            Dictionary with results summary
        """
        summary = {
            'total_results': 0,
            'results_by_status': {},
            'total_size': 0,
            'oldest_result': None,
            'newest_result': None
        }
        
        if not self.results_dir.exists():
            return summary
        
        total_size = 0
        oldest_time = None
        newest_time = None
        
        for result_dir in self.results_dir.iterdir():
            if result_dir.is_dir():
                summary['total_results'] += 1
                
                # Calculate directory size
                dir_size = sum(f.stat().st_size for f in result_dir.rglob('*') if f.is_file())
                total_size += dir_size
                
                # Track oldest and newest
                dir_time = result_dir.stat().st_mtime
                if oldest_time is None or dir_time < oldest_time:
                    oldest_time = dir_time
                    summary['oldest_result'] = result_dir.name
                
                if newest_time is None or dir_time > newest_time:
                    newest_time = dir_time
                    summary['newest_result'] = result_dir.name
        
        summary['total_size'] = total_size
        
        return summary
    
    def export_results_archive(self, output_path: Path, job_ids: List[str] = None) -> None:
        """
        Export results as an archive
        
        Args:
            output_path: Path for the archive file
            job_ids: Specific job IDs to export (None for all)
        """
        import tarfile
        
        if job_ids is None:
            # Export all results
            job_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        else:
            # Export specific jobs
            job_dirs = [self.results_dir / job_id for job_id in job_ids 
                       if (self.results_dir / job_id).exists()]
        
        if not job_dirs:
            logger.warning("No results to export")
            return
        
        with tarfile.open(output_path, 'w:gz') as tar:
            for job_dir in job_dirs:
                tar.add(job_dir, arcname=job_dir.name)
        
        logger.info(f"Exported {len(job_dirs)} result directories to {output_path}")

