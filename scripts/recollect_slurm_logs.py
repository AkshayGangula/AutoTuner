#!/usr/bin/env python3
"""
Backfill SLURM stdout/stderr into results/<job_id>/ using the same glob + sacct logic as
SLURMResultCollector (for experiments where logs were written outside the collector's search path).

Usage (on the login node, from repo root):
  python3 scripts/recollect_slurm_logs.py data/experiments/hybrid_vec_<id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autotuner.automation.slurm.result_collector import SLURMResultCollector  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/recollect_slurm_logs.py <experiment_dir>", file=sys.stderr)
        sys.exit(2)
    exp = Path(sys.argv[1]).expanduser().resolve()
    pid_file = exp / "phase2_job_ids.json"
    if not pid_file.is_file():
        print(f"No {pid_file}", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(pid_file.read_text())
    job_ids = [str(x) for x in raw] if isinstance(raw, list) else []
    rc = SLURMResultCollector(exp)
    for jid in job_ids:
        rd = exp / "results" / jid
        rd.mkdir(parents=True, exist_ok=True)
        rc._copy_slurm_files(jid, rd)
        rc._create_results_summary(jid, rd)
        print(f"Updated results/{jid}/ (SLURM logs where found)")


if __name__ == "__main__":
    main()
