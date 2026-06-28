#!/usr/bin/env python3
"""
SLURM Job Monitoring and Status Management

Handles job submission, monitoring, and status checking for SLURM jobs.
"""

import os
import subprocess
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

from autotuner.utils.path_utils import get_logical_cwd
from .models import JobStatus

logger = logging.getLogger(__name__)


class SLURMJobMonitor:
    """Monitors and manages SLURM job execution"""
    
    def __init__(self, work_directory):
        """
        Initialize the job monitor
        
        Args:
            work_directory: Working directory for job management
        """
        self.work_directory = work_directory
        self.job_history: Dict[str, Dict[str, any]] = {}
        self.active_jobs: Dict[str, Dict[str, any]] = {}
    
    def submit_job(self, script_path, submit_cwd=None) -> str:
        """
        Submit a SLURM job
        
        Args:
            script_path: Path to the job script
            submit_cwd: If set, run ``sbatch`` with this cwd (scratch submit dir on LEAP2-style sites)
            
        Returns:
            Job ID of the submitted job
        """
        try:
            script_path_obj = Path(script_path)
            if script_path_obj.is_absolute():
                absolute_script_path = script_path_obj
            else:
                base = get_logical_cwd()
                absolute_script_path = base / script_path_obj
            
            # Verify file exists
            if not absolute_script_path.exists():
                error_msg = f"Script file does not exist: {absolute_script_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            run_kw: dict = {"capture_output": True, "text": True}
            if submit_cwd is not None:
                run_kw["cwd"] = str(Path(submit_cwd))
                logger.info("Submitting sbatch from scratch cwd: %s", submit_cwd)
            # Submit job with absolute path
            result = subprocess.run(
                ["sbatch", str(absolute_script_path)],
                **run_kw,
            )
            
            if result.returncode != 0:
                logger.error(f"Job submission failed: {result.stderr}")
                raise RuntimeError(f"Job submission failed: {result.stderr}")
            
            # Extract job ID
            job_id = self._extract_job_id(result.stdout)
            if not job_id:
                raise RuntimeError("Could not extract job ID from submission output")
            
            # Record job
            self.active_jobs[job_id] = {
                'script_path': str(absolute_script_path),
                'submit_time': time.time(),
                'status': 'PENDING'
            }
            
            logger.info(f"Submitted job {job_id} using script {absolute_script_path}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise
    
    def _extract_job_id(self, output: str) -> Optional[str]:
        """Extract job ID from sbatch output"""
        match = re.search(r'Submitted batch job (\d+)', output)
        return match.group(1) if match else None
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the status of a SLURM job
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            JobStatus object or None if not found
        """
        try:
            # Use simpler squeue format that works on all SLURM systems
            # Uses subprocess.PIPE instead of capture_output for better compatibility
            result = subprocess.run(
                ["squeue", "-j", job_id, "--format=%i|%T|%M", "--noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Job is still in queue or running
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split('|')
                        if len(parts) >= 3 and parts[0].strip() == job_id:
                            return JobStatus(
                                job_id=parts[0].strip(),
                                state=parts[1].strip(),
                                submit_time="",
                                start_time="",
                                end_time="",
                                runtime=parts[2].strip() if len(parts) > 2 else "",
                                exit_code="",
                                nodes="",
                                cpus="",
                                memory=""
                            )
            
            # Job not in squeue - might have completed, check with sacct
            return self._check_completed_job(job_id)
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout checking status for job {job_id}")
            return None
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return None
    
    def _check_completed_job(self, job_id: str) -> Optional[JobStatus]:
        """Check if job has completed using sacct"""
        try:
            # Use compatible arguments for older Python versions
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "--noheader", "--parsable2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split('|')
                        if len(parts) >= 3:
                            job_id_part = parts[0].strip()
                            # Match job ID (might have .batch suffix)
                            if job_id_part.startswith(job_id):
                                state = parts[1].strip()
                                exit_code = parts[2].strip() if len(parts) > 2 else ""
                                
                                return JobStatus(
                                    job_id=job_id,
                                    state=state,
                                    submit_time="",
                                    start_time="",
                                    end_time="",
                                    runtime="",
                                    exit_code=exit_code,
                                    nodes="",
                                    cpus="",
                                    memory=""
                                )
            return None
        except Exception as e:
            logger.debug(f"Error checking completed job {job_id}: {e}")
            return None
    
    def monitor_jobs(self, job_ids: List[str], check_interval: int = 30) -> Dict[str, str]:
        """
        Monitor multiple jobs until completion
        
        Args:
            job_ids: List of job IDs to monitor
            check_interval: Check interval in seconds
            
        Returns:
            Dictionary mapping job IDs to final states
        """
        logger.info(f"Monitoring {len(job_ids)} jobs...")
        
        job_states = {}
        active_jobs = job_ids.copy()
        
        while active_jobs:
            for job_id in active_jobs[:]:
                status = self.get_job_status(job_id)
                
                if status:
                    current_state = status.state
                    job_states[job_id] = current_state
                    
                    # Check if job is complete
                    if current_state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                        logger.info(f"Job {job_id} completed with state: {current_state}")
                        if current_state == 'TIMEOUT':
                            out_path = Path(self.work_directory) / f"slurm-{job_id}.out"
                            err_path = Path(self.work_directory) / f"slurm-{job_id}.err"
                            logger.warning(
                                f"Job {job_id} timed out. Check {out_path} and {err_path} to see where the job was stuck."
                            )
                        active_jobs.remove(job_id)
                        
                        # Update job history
                        if job_id in self.active_jobs:
                            self.active_jobs[job_id]['status'] = current_state
                            self.active_jobs[job_id]['end_time'] = time.time()
                            self.job_history[job_id] = self.active_jobs.pop(job_id)
                    else:
                        logger.debug(f"Job {job_id} status: {current_state}")
                else:
                    logger.warning(f"Could not get status for job {job_id}")
            
            if active_jobs:
                logger.info(f"Waiting {check_interval}s... ({len(active_jobs)} jobs still active)")
                time.sleep(check_interval)
        
        logger.info("All jobs completed")
        return job_states
    
    def get_job_summary(self) -> Dict[str, any]:
        """
        Get summary of all jobs
        
        Returns:
            Dictionary with job summary information
        """
        summary = {
            'total_jobs': len(self.job_history) + len(self.active_jobs),
            'completed_jobs': len(self.job_history),
            'active_jobs': len(self.active_jobs),
            'job_states': {}
        }
        
        # Count job states
        for job_data in self.job_history.values():
            state = job_data.get('status', 'UNKNOWN')
            summary['job_states'][state] = summary['job_states'].get(state, 0) + 1
        
        for job_data in self.active_jobs.values():
            state = job_data.get('status', 'UNKNOWN')
            summary['job_states'][state] = summary['job_states'].get(state, 0) + 1
        
        return summary
    
    def cleanup_jobs(self, job_ids: List[str]) -> None:
        """
        Clean up completed jobs
        
        Args:
            job_ids: List of job IDs to clean up
        """
        logger.info(f"Cleaning up {len(job_ids)} jobs...")
        
        for job_id in job_ids:
            try:
                # Remove from active jobs if present
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                
                # Remove from job history if present
                if job_id in self.job_history:
                    del self.job_history[job_id]
                
                # Clean up SLURM output files
                slurm_files = list(self.work_directory.glob(f"slurm-{job_id}.*"))
                for slurm_file in slurm_files:
                    slurm_file.unlink()
                
                logger.info(f"Cleaned up job {job_id}")
                
            except Exception as e:
                logger.error(f"Error cleaning up job {job_id}: {e}")
    
    def export_job_history(self, output_path) -> None:
        """
        Export job history to files
        
        Args:
            output_path: Directory to save exported data
        """
        import json
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export job history
        history_file = output_path / "job_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.job_history, f, indent=2, default=str)
        
        # Export active jobs
        active_file = output_path / "active_jobs.json"
        with open(active_file, 'w') as f:
            json.dump(self.active_jobs, f, indent=2, default=str)
        
        # Export summary
        summary_file = output_path / "job_summary.json"
        summary = self.get_job_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Exported job data to {output_path}")

