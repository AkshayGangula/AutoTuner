"""
Parse application stdout embedded in SLURM / run1 logs (LULESH, miniMD, SPARSE/HYBRID_VEC_GPU lines).

App-specific parsers live in ``autotuner.app_registry.lulesh`` and ``autotuner.app_registry.minimd``;
this module composes them for generic SLURM / job output.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autotuner.app_registry.lulesh import (
    lulesh_throughput_z_per_s_from_mesh,
    parse_lulesh_mesh_iter_from_job_script,
    parse_lulesh_runtime_throughput,
)
from autotuner.app_registry.minimd import parse_minimd_runtime_throughput

logger = logging.getLogger(__name__)

_RE_TIME = re.compile(r"Time=([\d.]+)\s*(?:sec|seconds)?", re.IGNORECASE)
_RE_THROUGHPUT = re.compile(r"Throughput=([\d.]+)\s*(?:GFLOPS)?", re.IGNORECASE)


def parse_slurm_file_runtime_throughput(slurm_file: Path) -> Tuple[float, float]:
    """Runtime (sec) and throughput from a SLURM output file (Run 1a, SPARSE/HYBRID_VEC lines, LULESH, miniMD, regex)."""
    runtime = 0.0
    throughput = 0.0
    try:
        content = slurm_file.read_text(errors="replace")
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        lines = content.split("\n")
        killed = (
            "CANCELLED" in content
            or "Job step aborted" in content
            or ": Killed" in content
        )

        for line in lines:
            if line.startswith("Run 1a Time:"):
                try:
                    val = float(line.split(":")[1].replace("seconds", "").strip())
                    if val > 0:
                        runtime = val
                        break
                except (ValueError, IndexError):
                    pass

        if runtime > 0:
            in_run1a = False
            for line in lines:
                if line.startswith("Run 1a") or "Starting Run 1a" in line:
                    in_run1a = True
                if in_run1a and line.startswith("Run 1a FOM:"):
                    try:
                        val = float(line.split(":", 1)[1].strip().split()[0])
                        if val > 0:
                            throughput = val
                    except (ValueError, IndexError):
                        pass
                if in_run1a and (
                    "SPARSE:" in line
                    or "HYBRID_VEC_GPU:" in line
                ):
                    for part in line.split(","):
                        if "Throughput=" in part:
                            try:
                                val = float(part.split("=")[1].replace(" GFLOPS", "").strip())
                                if val > 0:
                                    throughput = val
                            except (ValueError, IndexError):
                                pass
                    if throughput > 0:
                        break
                if in_run1a and ("Starting Run 1b" in line or "Starting Run 2" in line):
                    break
            if throughput <= 0:
                run1a_chunk: List[str] = []
                in_run1a = False
                for line in lines:
                    if line.startswith("Run 1a") or "Starting Run 1a" in line:
                        in_run1a = True
                    if in_run1a:
                        run1a_chunk.append(line)
                    if in_run1a and ("Starting Run 1b" in line or "Starting Run 2" in line):
                        break
                chunk_txt = "\n".join(run1a_chunk)
                _, lt = parse_lulesh_runtime_throughput(chunk_txt)
                if lt > 0:
                    throughput = lt
                if throughput <= 0:
                    _, mt = parse_minimd_runtime_throughput(chunk_txt)
                    if mt > 0:
                        throughput = mt
            return runtime, throughput

        for line in reversed(lines):
            if (
                "SPARSE:" in line
                or "HYBRID_VEC_GPU:" in line
            ):
                if "Time=" not in line:
                    continue
                parts = line.split(",")
                for part in parts:
                    if "Time=" in part:
                        try:
                            val = float(
                                part.split("=")[1]
                                .replace(" sec", "")
                                .replace(" seconds", "")
                                .strip()
                            )
                            if val > 0 or not killed:
                                runtime = val
                        except (ValueError, IndexError):
                            pass
                    if "Throughput=" in part:
                        try:
                            val = float(part.split("=")[1].replace(" GFLOPS", "").strip())
                            if val > 0 or not killed:
                                throughput = val
                        except (ValueError, IndexError):
                            pass
                if runtime > 0:
                    break

        if runtime <= 0 or throughput <= 0:
            lulesh_scope = (
                content.split("Starting Run 1b", 1)[0]
                if "Starting Run 1b" in content
                else content
            )
            lr, lt = parse_lulesh_runtime_throughput(lulesh_scope)
            if runtime <= 0 and lr > 0:
                runtime = lr
            if throughput <= 0 and lt > 0:
                throughput = lt

        if runtime <= 0 or throughput <= 0:
            minimd_scope = (
                content.split("Starting Run 1b", 1)[0]
                if "Starting Run 1b" in content
                else content
            )
            if "PERF_SUMMARY" in minimd_scope or "miniMD" in minimd_scope:
                mr, mt = parse_minimd_runtime_throughput(minimd_scope)
                if runtime <= 0 and mr > 0:
                    runtime = mr
                if throughput <= 0 and mt > 0:
                    throughput = mt

        if runtime <= 0 or throughput < 0:
            for m in _RE_TIME.finditer(content):
                try:
                    runtime = float(m.group(1))
                    if runtime > 0:
                        break
                except ValueError:
                    pass
            for m in _RE_THROUGHPUT.finditer(content):
                try:
                    throughput = float(m.group(1))
                    if throughput >= 0:
                        break
                except ValueError:
                    pass
    except Exception as e:
        logger.warning("Could not parse %s: %s", slurm_file, e)
    return runtime, throughput


def parse_slurm_job_wall_clock_seconds(content: str) -> float:
    """
    Last-resort runtime when the application never printed Time=/Run 1a Time (e.g. Exec format error).

    AutoTuner job scripts echo GNU ``date`` as ``Start time:`` / ``End time:``; the delta is **job
    wall clock** (Run 1a + LIKWID + wrappers), not pure GPU/CPU kernel time.
    """
    start_m = re.search(r"(?m)^Start time:\s*(.+)$", content)
    end_m = re.search(r"(?m)^End time:\s*(.+)$", content)
    if not start_m or not end_m:
        return 0.0
    s_raw = start_m.group(1).strip()
    e_raw = end_m.group(1).strip()

    def _parse_one(line: str) -> Optional[datetime]:
        for fmt in ("%a %b %d %H:%M:%S %Z %Y", "%a %b %e %H:%M:%S %Z %Y"):
            try:
                return datetime.strptime(line, fmt)
            except ValueError:
                continue
        parts = line.split()
        if len(parts) >= 6:
            # Drop timezone token (e.g. CDT) — naive parse for same-calendar-day jobs
            no_tz = " ".join(parts[:-2] + [parts[-1]])
            for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b %e %H:%M:%S %Y"):
                try:
                    return datetime.strptime(no_tz, fmt)
                except ValueError:
                    continue
        return None

    ts = _parse_one(s_raw)
    te = _parse_one(e_raw)
    if ts is None or te is None:
        return 0.0
    sec = (te - ts).total_seconds()
    if 0.001 < sec < 7 * 86400:
        return float(sec)
    return 0.0


def infer_hybrid_vec_gflops_from_slurm_cli(content: str, runtime_sec: float) -> float:
    """
    Rough GFLOPS from ``--size N`` / ``--iterations M`` in the SLURM log and hybrid_vec's
    ~3·N·M flop model (total elements × iterations). Order-of-magnitude when wall-clock is used.
    """
    if runtime_sec <= 0:
        return 0.0
    sm = re.search(r"--size\s+(\d+)", content, re.IGNORECASE)
    im = re.search(r"--iterations\s+(\d+)", content, re.IGNORECASE)
    if not sm or not im:
        return 0.0
    try:
        n = int(sm.group(1))
        it = int(im.group(1))
    except ValueError:
        return 0.0
    flops = 3.0 * float(n) * float(it)
    return flops / (runtime_sec * 1e9)


def parse_job_stdout_runtime_throughput(output: str) -> Tuple[float, float]:
    """Nsight fallback: last SPARSE/HYBRID_VEC_GPU line, then LULESH, then miniMD."""
    runtime, throughput = 0.0, 0.0
    killed = (
        "CANCELLED" in output
        or "Job step aborted" in output
        or ": Killed" in output
    )
    for line in reversed(output.split("\n")):
        if (
            "SPARSE:" not in line
            and "HYBRID_VEC_GPU:" not in line
        ):
            continue
        if "Time=" not in line:
            continue
        tm = re.search(r"Time=([\d.]+)\s*(?:sec|seconds)?", line)
        tp = re.search(r"Throughput=([\d.]+)\s*GFLOPS", line)
        if tm:
            val = float(tm.group(1))
            if val > 0 or not killed:
                runtime = val
        if tp:
            val = float(tp.group(1))
            if val > 0 or not killed:
                throughput = val
        if runtime > 0:
            break
    if runtime <= 0 or throughput <= 0:
        lr, lt = parse_lulesh_runtime_throughput(output)
        if runtime <= 0 and lr > 0:
            runtime = lr
        if throughput <= 0 and lt > 0:
            throughput = lt
    if runtime <= 0 or throughput <= 0:
        minimd_scope = (
            output.split("Starting Run 1b", 1)[0]
            if "Starting Run 1b" in output
            else output
        )
        if "PERF_SUMMARY" in minimd_scope or "miniMD" in minimd_scope:
            mr, mt = parse_minimd_runtime_throughput(minimd_scope)
            if runtime <= 0 and mr > 0:
                runtime = mr
            if throughput <= 0 and mt > 0:
                throughput = mt
    return runtime, throughput


def infer_stdout_metadata(output: str) -> Dict[str, Any]:
    """application_type, problem_size, iterations when present in stdout."""
    metrics: Dict[str, Any] = {}
    type_match = re.search(
        r"(SPARSE|HYBRID_VEC_GPU):", output
    )
    if type_match:
        metrics["application_type"] = type_match.group(1)
    elif "Run completed:" in output and re.search(r"Elapsed\s+time\s*=", output):
        metrics["application_type"] = "LULESH"
    elif "miniMD" in output or "PERF_SUMMARY" in output:
        metrics["application_type"] = "miniMD"

    size_match = re.search(r"Size=(\d+)", output)
    if size_match:
        metrics["problem_size"] = int(size_match.group(1))
    else:
        lulesh_nx = re.search(r"Problem\s+size\s*=\s*(\d+)", output)
        if lulesh_nx:
            metrics["problem_size"] = int(lulesh_nx.group(1))

    iterations_match = re.search(r"Iterations=(\d+)", output)
    if iterations_match:
        metrics["iterations"] = int(iterations_match.group(1))
    else:
        lulesh_it = re.search(r"Iteration\s+count\s*=\s*(\d+)", output)
        if lulesh_it:
            metrics["iterations"] = int(lulesh_it.group(1))
    return metrics