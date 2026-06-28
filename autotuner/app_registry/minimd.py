"""miniMD-specific stdout parsing and SLURM Run-1a timing hooks (Mantevo miniMD)."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def extract_minimd_input_clause(arguments: str) -> Optional[str]:
    """Return ``-i <path>`` or ``--input_file <path>`` substring from CLI, if present."""
    s = (arguments or "").strip()
    if not s:
        return None
    m = re.search(r"(?:^|\s)(-i|--input_file)(?:\s+|\s*=\s*)(\S+)", s)
    if not m:
        return None
    flag, path = m.group(1), m.group(2)
    return f"{flag} {path}"


def merge_minimd_phase1_cli(full_arguments: str, phase1_arguments: str) -> str:
    """
    Prepend input deck from full ``--arguments`` when ``--phase1-args`` omits ``-i``.

    Phase-1 calls the binary with ``phase1_arguments`` as the sole CLI; if that string
    is only ``-n/-t/...``, the default ``in.lj.miniMD`` may not exist on compute nodes
    and no PERF_SUMMARY line appears — Phase-1 parsing then yields no runtime.
    """
    p1 = (phase1_arguments or "").strip()
    if not p1:
        return p1
    if re.search(r"(?:^|\s)(-i|--input_file)(?:\s+|\s*=\s*)\S+", p1):
        return p1
    clause = extract_minimd_input_clause(full_arguments)
    if not clause:
        return p1
    return f"{clause} {p1}".strip()

# PERF_SUMMARY line layout matches ref/ljs.cpp printf: total wall time is 7 fields before PERF_SUMMARY.
_MINIMD_PERF_SUMMARY_TIME_FIELD_OFFSET = 7


def slurm_full_runtime_fallback_from_run1_out_lines() -> List[str]:
    """
    Bash fragment appended after hybrid_vec/LULESH grep fails: set FULL_RUNTIME from miniMD PERF_SUMMARY.

    Injected into generated SLURM scripts so ``Run 1a Time:`` appears in slurm-*.out for Phase-1 parsing.
    """
    off = _MINIMD_PERF_SUMMARY_TIME_FIELD_OFFSET
    return [
        "# miniMD (Mantevo): wall time = field %d before PERF_SUMMARY (ref/ljs.cpp); Phase-1 / slurm parsing."
        % off,
        "if [ -z \"$FULL_RUNTIME\" ]; then",
        "    FULL_RUNTIME=$(grep PERF_SUMMARY \"$RUN1_OUT_LOCAL\" 2>/dev/null | tail -1 | sed 's/^#//' | awk '",
        f'        {{ for (i = 1; i <= NF; i++) if ($i == "PERF_SUMMARY" && i > {off}) {{ print $(i - {off}); exit }} }}',
        "    ' || true)",
        "fi",
    ]


def parse_minimd_runtime_throughput(text: str) -> Tuple[float, float]:
    """
    Mantevo miniMD: last PERF_SUMMARY line in *text* wins. Strip leading '#'.
    """
    runtime = 0.0
    throughput = 0.0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        if "PERF_SUMMARY" not in s:
            continue
        parts = s.split()
        try:
            idx = parts.index("PERF_SUMMARY")
            if idx >= _MINIMD_PERF_SUMMARY_TIME_FIELD_OFFSET:
                runtime = float(parts[idx - _MINIMD_PERF_SUMMARY_TIME_FIELD_OFFSET])
            if idx >= 2:
                throughput = float(parts[idx - 2])
        except (ValueError, IndexError):
            continue
    return runtime, throughput
