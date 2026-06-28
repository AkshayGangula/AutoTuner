#!/usr/bin/env python3
"""
Phase-1 subset selection for "Profiling only a subset" two-phase auto-tuning.

Phase-1 preliminary screening collects: runtime, throughput (GFLOPS), and LIKWID locality when possible.
Selection uses: Stratified (top-1 or top-2 by runtime per stratum), Pareto front (runtime, throughput;
runtime, locality if available), Near-best (within X% of best runtime), Diversity (1-2 configs
maximizing distance from already-selected).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from autotuner.utils.path_utils import (
    experiment_artifact_dir_paths,
    experiment_path_roots,
    find_slurm_log_for_job,
    get_logical_cwd,
)
from autotuner.app_registry.stdout_parsing import parse_slurm_file_runtime_throughput as _parse_slurm_output

logger = logging.getLogger(__name__)


def get_stratum_key(mpi_ranks: int) -> str:
    """
    Map MPI rank count to stratum key for stratified selection.
    Strata: 1, 2-4, 8-16, 32+.
    """
    if mpi_ranks <= 1:
        return "1"
    if mpi_ranks <= 4:
        return "2-4"
    if mpi_ranks <= 16:
        return "8-16"
    return "32+"


def collect_phase1_data(
    work_directory: Path,
    job_results: Dict[str, Any],
    config_map: Dict[str, Any],
    system_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Collect Phase-1 preliminary screening data: runtime, throughput, locality (when LIKWID available).

    Args:
        work_directory: Experiment work directory (SLURM output and likwid_output live here).
        job_results: config_name -> {job_id, status, results_directory?, config, ...}.
        config_map: config_name -> config object (for mpi_ranks_per_node, etc.).

    Returns:
        config_name -> {runtime, throughput, locality, config, job_id, stratum_key}.
    """
    wd = Path(work_directory)
    base = get_logical_cwd()
    work_directory = (base / wd) if not wd.is_absolute() else wd
    phase1_data = {}
    for config_name, result in job_results.items():
        if result.get("status") != "COMPLETED" or "job_id" not in result:
            continue
        job_id = result["job_id"]
        config = result.get("config") or config_map.get(config_name)
        if not config:
            continue

        # Runtime/throughput: parse application-ish logs only (not LIKWID CSV — _parse_slurm_output
        # would always return 0 and used to make Phase-1 miss LULESH/hybrid_vec timings).
        # Order: run1_*.out (tee of Run 1a), slurm-*.out, then slurm-*.err.
        runtime, throughput = 0.0, 0.0
        jid = str(job_id)
        candidates: List[Path] = []

        def _add(p: Path) -> None:
            if p not in candidates:
                candidates.append(p)

        res_dir: Optional[Path] = None
        if result.get("results_directory"):
            rd = Path(result["results_directory"])
            res_dir = (get_logical_cwd() / rd) if not rd.is_absolute() else rd
            if not res_dir.is_dir():
                for alt in experiment_path_roots(res_dir):
                    if alt.is_dir():
                        res_dir = alt
                        break
                else:
                    res_dir = None

        if res_dir is not None:
            _add(res_dir / f"run1_{jid}.out")
            _add(res_dir / f"slurm-{jid}.out")
            _add(res_dir / f"slurm-{jid}.err")
        for root in experiment_path_roots(work_directory):
            _add(root / "results" / jid / f"run1_{jid}.out")
            _add(root / f"run1_{jid}.out")
            _add(root / "results" / jid / f"slurm-{jid}.out")
            _add(root / f"slurm-{jid}.out")
            _add(root / "results" / jid / f"slurm-{jid}.err")
            _add(root / f"slurm-{jid}.err")
        for root in experiment_artifact_dir_paths(work_directory, system_config):
            _add(root / f"run1_{jid}.out")
            _add(root / "likwid_output" / jid / f"likwid_0.csv")
        _slurm_discovered = find_slurm_log_for_job(
            work_directory, jid, system_config=system_config
        )
        if _slurm_discovered is not None:
            _add(_slurm_discovered)
            if _slurm_discovered.suffix.lower() == ".out":
                err_sibling = _slurm_discovered.with_suffix(".err")
                if err_sibling.is_file():
                    _add(err_sibling)

        slurm_file: Optional[Path] = None
        preferred_hint: Optional[Path] = None  # first .out log for clearer warnings
        for p in candidates:
            if not p.exists() or not p.is_file():
                continue
            if preferred_hint is None and p.suffix.lower() == ".out":
                preferred_hint = p
            slurm_file = p
            runtime, throughput = _parse_slurm_output(p)
            if runtime > 0:
                break
        if runtime <= 0:
            path_hint = preferred_hint or slurm_file or (candidates[0] if candidates else None)
            if path_hint and path_hint.exists():
                logger.debug(
                    "Phase-1: no runtime parsed from %s (expect Run 1a Time:, "
                    "HYBRID_VEC_GPU/SPARSE Time=, LULESH Elapsed, or miniMD PERF_SUMMARY)",
                    path_hint,
                )
                try:
                    snippet = path_hint.read_text(errors="replace")[:8000]
                except OSError:
                    snippet = ""
                diag = ""
                if "Exec format error" in snippet or "cannot execute binary file" in snippet:
                    diag = " (binary/exec mismatch on compute nodes — rebuild for the GPU node's architecture)"
                elif "error while loading shared libraries" in snippet:
                    diag = " (missing shared library on compute nodes)"
                elif "CUDA driver version is insufficient" in snippet or "no CUDA-capable device" in snippet:
                    diag = " (CUDA/device error on compute nodes)"
                logger.warning(
                    "Phase-1: no runtime for %s (job %s) — no timing lines in logs%s; see %s",
                    config_name,
                    job_id,
                    diag,
                    path_hint,
                )
            else:
                logger.warning(
                    "Phase-1: no runtime for %s (job %s) — no log files found (checked run1/slurm paths)",
                    config_name,
                    job_id,
                )
            continue

        locality_value = 0.0
        try:
            from autotuner.core.likwid_profiler import get_locality_for_job

            locality_value, _src = get_locality_for_job(work_directory, str(job_id))
        except Exception as e:
            logger.debug("Phase-1 locality from LIKWID failed for job %s: %s", job_id, e)

        # Strata use total MPI ranks (nodes × ranks/node), not ranks-per-node alone.
        _nodes = int(getattr(config, "num_nodes", None) or 1)
        _rpn = int(getattr(config, "mpi_ranks_per_node", 1) or 1)
        total_mpi_ranks = max(1, _nodes * _rpn)
        stratum_key = get_stratum_key(total_mpi_ranks)

        phase1_data[config_name] = {
            "runtime": runtime,
            "throughput": throughput,
            "locality": locality_value,
            "config": config,
            "job_id": job_id,
            "stratum_key": stratum_key,
        }
    return phase1_data


def stratified_selection(
    phase1_data: Dict[str, Dict[str, Any]],
    configs: List[Any],
    top_per_stratum: int = 2,
) -> Set[str]:
    """
    Select top top_per_stratum configs by runtime within each stratum.
    """
    from collections import defaultdict

    by_stratum = defaultdict(list)
    for config_name, data in phase1_data.items():
        by_stratum[data["stratum_key"]].append((config_name, data["runtime"]))
    selected = set()
    for stratum_key, items in by_stratum.items():
        items.sort(key=lambda x: x[1])  # by runtime ascending
        for i, (name, _) in enumerate(items[:top_per_stratum]):
            selected.add(name)
    return selected


def pareto_front(
    phase1_data: Dict[str, Dict[str, Any]],
    objectives: List[Tuple[str, bool]],
) -> Set[str]:
    """
    objectives: list of (metric_name, minimize).
    E.g. [("runtime", True), ("throughput", False)] -> minimize runtime, maximize throughput.
    Returns set of config names that are non-dominated.
    """
    names = list(phase1_data.keys())
    if not names:
        return set()
    dominated = set()
    for i, name_i in enumerate(names):
        if name_i in dominated:
            continue
        data_i = phase1_data[name_i]
        for j, name_j in enumerate(names):
            if i == j or name_j in dominated:
                continue
            data_j = phase1_data[name_j]
            # Check if j dominates i: j no worse on all objectives and strictly better on at least one
            worse = False
            better = False
            for metric, minimize in objectives:
                vi = data_i.get(metric, 0.0)
                vj = data_j.get(metric, 0.0)
                if minimize:
                    if vj < vi:
                        better = True
                    elif vj > vi:
                        worse = True
                else:
                    if vj > vi:
                        better = True
                    elif vj < vi:
                        worse = True
            if better and not worse:
                dominated.add(name_i)
                break
    return set(n for n in names if n not in dominated)


def near_best_selection(
    phase1_data: Dict[str, Dict[str, Any]],
    already_selected: Set[str],
    pct: float = 10.0,
) -> Set[str]:
    """Add configs whose runtime is within pct% of the best runtime (across all configs)."""
    if not phase1_data:
        return set()
    best_runtime = min(data["runtime"] for data in phase1_data.values())
    threshold = best_runtime * (1 + pct / 100.0)
    added = set()
    for name, data in phase1_data.items():
        if name in already_selected:
            continue
        if data["runtime"] <= threshold:
            added.add(name)
    return added


def diversity_selection(
    phase1_data: Dict[str, Dict[str, Any]],
    configs: List[Any],
    already_selected: Set[str],
    num_slots: int = 2,
) -> Set[str]:
    """
    Add up to num_slots configs that maximize distance from already_selected.
    Distance = L2 in (mpi_ranks_per_node, omp_threads_per_rank) normalized by max.
    """
    if num_slots <= 0 or not configs:
        return set()
    config_by_name = {c.name: c for c in configs}
    selected_list = [
        (config_by_name[n].mpi_ranks_per_node, config_by_name[n].omp_threads_per_rank)
        for n in already_selected
        if n in config_by_name
    ]
    if not selected_list:
        # Pick first num_slots by some default (e.g. spread in mpi_ranks)
        remaining = [c for c in configs if c.name in phase1_data and c.name not in already_selected]
        remaining.sort(key=lambda c: (c.mpi_ranks_per_node, c.omp_threads_per_rank))
        return set(c.name for c in remaining[:num_slots])

    max_r = max(c.mpi_ranks_per_node for c in configs)
    max_t = max(c.omp_threads_per_rank for c in configs)
    if max_r == 0:
        max_r = 1
    if max_t == 0:
        max_t = 1

    def dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return ((a[0] / max_r - b[0] / max_r) ** 2 + (a[1] / max_t - b[1] / max_t) ** 2) ** 0.5

    candidates = [
        c for c in configs
        if c.name in phase1_data and c.name not in already_selected
    ]
    added = set()
    for _ in range(num_slots):
        if not candidates:
            break
        best_name = None
        best_min_dist = -1.0
        for c in candidates:
            if c.name in added:
                continue
            pt_c = (c.mpi_ranks_per_node, c.omp_threads_per_rank)
            min_d = min(dist(pt_c, s) for s in selected_list)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_name = c.name
        if best_name is not None:
            added.add(best_name)
            c = config_by_name[best_name]
            selected_list.append((c.mpi_ranks_per_node, c.omp_threads_per_rank))
            candidates = [c for c in candidates if c.name != best_name]
    return added


def select_subset(
    phase1_data: Dict[str, Dict[str, Any]],
    configs: List[Any],
    stratum_top_n: int = 2,
    near_best_pct: float = 10.0,
    diversity_slots: int = 2,
    max_runtime_multiplier: float = 2.0,
) -> List[str]:
    """
    Combined selection: Hard cutoff → Stratified + Pareto + Near-best + Diversity.
    Returns ordered list of config names to run in Phase-2 (full profiling).

    Args:
        max_runtime_multiplier: Hard pre-filter - discard any config whose Phase-1
            runtime exceeds best_runtime * max_runtime_multiplier BEFORE any other
            selection strategy runs. Set to 0 or None to disable.
            Example: 2.0 → drop configs that are >2× slower than the fastest config.
    """
    if not phase1_data:
        return []

    # ── Step 0: Hard runtime cutoff ────────────────────────────────────────────
    # Eliminate configs that are clearly too slow before strategies run.
    # This prevents diversity or Pareto from rescuing a 5× slower config.
    eligible_data = phase1_data
    if max_runtime_multiplier and max_runtime_multiplier > 0:
        best_runtime = min(d["runtime"] for d in phase1_data.values() if d["runtime"] > 0)
        cutoff = best_runtime * max_runtime_multiplier
        elapsed_param = "runtime" # Use runtime for cutoff

        # Topology Safeguard: Identify canonical architectures to protect
        # 1. Pure MPI (Max Ranks)
        # 2. Pure OpenMP (Min Ranks, usually 1)
        # 3. Socket/NUMA-aligned (2 or 4 Ranks)
        # These often map to physical hardware boundaries and are structurally important 
        # even if they suffer from cache penalties (vs RAM) at small Phase-1 sizes.
        max_ranks = max((c.mpi_ranks_per_node for c in configs), default=1)
        safelist = set()
        for c in configs:
            if c.mpi_ranks_per_node == max_ranks:
                safelist.add(c.name) # Pure MPI
            if c.mpi_ranks_per_node == 1:
                safelist.add(c.name) # Pure OpenMP
            if c.mpi_ranks_per_node in [2, 4]:
                safelist.add(c.name) # Socket/NUMA aligned
        
        eliminated = {n for n, d in phase1_data.items() if d["runtime"] > cutoff}
        
        # Apply Safeguard
        rescued = eliminated.intersection(safelist)
        if rescued:
             logger.info(
                 f"Phase-1 Safeguard: Rescued {len(rescued)} canonical configs from elimination: {sorted(rescued)} "
                 f"(likely memory-bound configs unfairly penalized by cache-resident winners)"
             )
             eliminated -= rescued

        if eliminated:
            logger.info(
                f"Phase-1 hard cutoff ({max_runtime_multiplier}× best={best_runtime:.3f}s, "
                f"threshold={cutoff:.3f}s): eliminated {len(eliminated)} config(s): {sorted(eliminated)}"
            )
            eligible_data = {n: d for n, d in phase1_data.items() if n not in eliminated}
        else:
            logger.info(
                f"Phase-1 hard cutoff ({max_runtime_multiplier}× best={best_runtime:.3f}s): "
                f"all configs within threshold (or rescued) — none eliminated"
            )
    else:
        logger.info("Phase-1 hard cutoff: disabled (max_runtime_multiplier=0)")

    if not eligible_data:
        # Cutoff was too aggressive — fall back to all configs to avoid empty Phase-2
        logger.warning(
            "Phase-1 hard cutoff eliminated ALL configs — falling back to full set."
        )
        eligible_data = phase1_data

    selected = set()

    # 1. Stratified: top stratum_top_n by runtime per stratum
    stratified = stratified_selection(eligible_data, configs, top_per_stratum=stratum_top_n)
    selected.update(stratified)
    logger.info(f"Phase-1 selection: Stratified added {len(stratified)} configs: {sorted(stratified)}")

    # 2. Pareto front (runtime minimize, throughput maximize)
    objectives_rt = [("runtime", True), ("throughput", False)]
    pareto_rt = pareto_front(eligible_data, objectives_rt)
    added_pareto_rt = pareto_rt - selected
    selected.update(pareto_rt)
    if added_pareto_rt:
        logger.info(f"Phase-1 selection: Pareto (runtime,throughput) added {len(added_pareto_rt)}: {sorted(added_pareto_rt)}")

    # 3. Pareto (runtime, locality) if any has locality
    for d in eligible_data.values():
        d.setdefault("locality", d.get("locality_estimate", 0.0))
    if any(d.get("locality", 0) > 0 for d in eligible_data.values()):
        objectives_loc = [("runtime", True), ("locality", False)]
        pareto_loc = pareto_front(eligible_data, objectives_loc)
        added_pareto_loc = pareto_loc - selected
        selected.update(pareto_loc)
        if added_pareto_loc:
            logger.info(f"Phase-1 selection: Pareto (runtime,locality) added {len(added_pareto_loc)}: {sorted(added_pareto_loc)}")

    # 4. Near-best: within near_best_pct% of best runtime
    near_best = near_best_selection(eligible_data, selected, pct=near_best_pct)
    selected.update(near_best)
    if near_best:
        logger.info(f"Phase-1 selection: Near-best (within {near_best_pct}%) added {len(near_best)}: {sorted(near_best)}")

    # 5. Diversity: up to diversity_slots (only from eligible configs)
    diversity = diversity_selection(eligible_data, configs, selected, num_slots=diversity_slots)
    selected.update(diversity)
    if diversity:
        logger.info(f"Phase-1 selection: Diversity added {len(diversity)}: {sorted(diversity)}")

    # Return as list, ordered by original config order (so report order is stable)
    name_to_order = {c.name: i for i, c in enumerate(configs)}
    return sorted(selected, key=lambda n: name_to_order.get(n, 999))


# Config name (e.g. 1x72) from "Starting job: ... 1x72" or "job-name: ... 1x72"
_CONFIG_PATTERN = re.compile(r"(\d+x\d+)")


def backfill_phase1_throughput(experiment_dir: Path) -> bool:
    """
    Re-parse Phase-1 SLURM outputs (HYBRID_VEC_GPU/SPARSE lines) and update
    phase1_results.json so that throughput (and runtime) are filled for existing runs.

    Returns True if phase1_results.json was updated, False otherwise.
    """
    experiment_dir = Path(experiment_dir).resolve()
    phase1_file = experiment_dir / "phase1_results.json"
    if not phase1_file.exists():
        return False
    try:
        data = json.loads(phase1_file.read_text())
    except Exception as e:
        logger.warning(f"Could not load {phase1_file}: {e}")
        return False
    phase1_data = data.get("phase1_data", {})
    if not phase1_data:
        return False

    # Find all slurm-*.out in experiment dir
    slurm_files = sorted(experiment_dir.glob("slurm-*.out"))
    # Build: config_name -> [(path, is_phase1), ...]; prefer Phase-1 (Iterations=2)
    config_to_candidates: Dict[str, List[Tuple[Path, bool]]] = {}
    for path in slurm_files:
        try:
            text = path.read_text()
        except Exception:
            continue
        lines = text.splitlines()
        config_name = None
        for line in lines[:80]:
            if "Starting job:" in line or "job-name:" in line.lower() or "job_name:" in line.lower():
                m = _CONFIG_PATTERN.search(line)
                if m:
                    config_name = m.group(1)
                    break
        if not config_name or config_name not in phase1_data:
            continue
        is_phase1 = "Iterations=2" in text or "iterations 2" in text.lower()
        config_to_candidates.setdefault(config_name, []).append((path, is_phase1))

    updated = False
    for config_name, candidates in config_to_candidates.items():
        # Prefer Phase-1 runs (light args)
        candidates.sort(key=lambda x: (not x[1], x[0].name))
        path = candidates[0][0]
        runtime, throughput = _parse_slurm_output(path)
        if throughput > 0 or runtime > 0:
            entry = phase1_data[config_name]
            if throughput > 0 and entry.get("throughput", 0) == 0:
                entry["throughput"] = throughput
                updated = True
            if runtime > 0:
                entry["runtime"] = runtime
    if not updated:
        return False
    try:
        with open(phase1_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Backfilled Phase-1 throughput (and runtime) in {phase1_file}")
        return True
    except Exception as e:
        logger.warning(f"Could not write {phase1_file}: {e}")
        return False
