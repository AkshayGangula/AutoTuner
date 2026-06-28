"""Shared path utilities for HPC environments."""
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Repository root (AutoTuner/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Current layout: data/experiments/<app>_<timestamp>/
EXPERIMENTS_REL = Path("data/experiments")
# Legacy path kept for discovery only (pre-restructure clones)
LEGACY_EXPERIMENTS_REL = Path("auto_tuning_experiments")


def experiment_collection_roots(base_dir: Optional[Path] = None) -> List[Path]:
    """
    Directories to search when resolving ``latest`` or a bare experiment folder name.
    Includes ``data/experiments`` and the legacy ``auto_tuning_experiments`` path.
    """
    root = Path(base_dir) if base_dir is not None else REPO_ROOT
    candidates = [
        root / EXPERIMENTS_REL,
        Path.cwd() / EXPERIMENTS_REL,
        root / LEGACY_EXPERIMENTS_REL,
        Path.cwd() / LEGACY_EXPERIMENTS_REL,
        Path.cwd(),
    ]
    out: List[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def get_logical_cwd() -> Path:
    """Current directory as the user sees it (preserves /home, avoids backing filesystem mounts)."""
    return Path(os.environ.get("PWD", os.getcwd()))


def normalize_to_logical_path(path: Union[str, Path]) -> Path:
    """
    Convert resolved/backing paths to the logical path users see.
    
    On HPC systems, /home is often the logical path users see, while the backing
    filesystem may be mounted at a different location (e.g., /mmfs1, /xfs, etc.).
    
    This function normalizes such paths so scripts and logs show the path users
    expect (and that works correctly on compute nodes).
    """
    p = Path(path)
    try:
        s = p.as_posix()
    except Exception:
        s = str(path)
    # Common backing mounts that resolve to /home
    for prefix in ("/mmfs1/home", "/mmfs/home", "/xfs/home"):
        if s.startswith(prefix + "/") or s == prefix:
            return Path("/home" + s[len(prefix):])
    return Path(s)


def experiment_path_roots(work_directory: Path) -> List[Path]:
    """
    Return ``work_directory`` plus a common alias when the same home tree is visible
    under both /home/... and /mmfs1/home/... (or /mmfs/home, /xfs/home).

    Phase-1 log collection uses PWD-joined paths while SLURM may record submit paths
    on the backing mount; trying both avoids false 'missing' logs when the paths
    are equivalent on disk.
    """
    wd = Path(work_directory).expanduser()
    out: List[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = p.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(p)

    add(wd)
    pos = wd.as_posix()
    for prefix in ("/mmfs1/home/", "/mmfs/home/", "/xfs/home/"):
        if pos.startswith(prefix):
            tail = pos[len(prefix) :].lstrip("/")
            if tail:
                add(Path("/home") / tail)
            return out
    if pos.startswith("/home/"):
        for bp in ("/mmfs1/home/", "/mmfs/home/", "/xfs/home/"):
            add(Path(bp.rstrip("/")) / pos[len("/home/") :].lstrip("/"))
    return out


def slurm_template_to_shell(template: str) -> str:
    """Turn Slurm path templates into bash-friendly tokens (``%u`` → ``$USER``)."""
    s = (template or "").strip()
    return s.replace("%u", "$USER").replace("%U", "$USER")


def slurm_template_to_host_path(template: str) -> Path:
    """Expand ``%u`` for login-node file existence checks (``sacct``, log collection)."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    s = (template or "").strip().replace("%U", user).replace("%u", user)
    return Path(s).expanduser()


def path_to_shell_home_literal(path: Optional[str]) -> str:
    """
    Emit a bash path literal using ``$HOME`` when ``path`` lies under the login user's home.

    Avoids hard-coding ``/home/user`` vs ``/mmfs1/home/user`` in generated job scripts.
    """
    if not path or not str(path).strip():
        return ""
    raw = str(path).strip()
    try:
        resolved = Path(raw).expanduser().resolve()
    except OSError:
        resolved = Path(raw).expanduser()
    pos = resolved.as_posix()
    for prefix in ("/mmfs1/home/", "/mmfs/home/", "/xfs/home/"):
        if pos.startswith(prefix):
            rel = pos[len(prefix) :].lstrip("/")
            if rel:
                return f'"$HOME/{rel}"'
            return '"$HOME"'
    candidates: List[Path] = []
    try:
        candidates.append(Path.home().resolve())
    except OSError:
        candidates.append(Path.home())
    candidates.append(normalize_to_logical_path(Path.home()))
    for home in candidates:
        try:
            home_r = home.resolve()
        except OSError:
            home_r = home
        try:
            if resolved == home_r or str(resolved).startswith(str(home_r) + os.sep):
                rel = resolved.relative_to(home_r).as_posix()
                if rel:
                    return f'"$HOME/{rel}"'
                return '"$HOME"'
        except ValueError:
            continue
    escaped = raw.replace('"', '\\"')
    return f'"{escaped}"'


def experiment_artifact_dir_shell_expr(
    work_directory: Union[str, Path],
    system_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Shell ``EXPERIMENT_DIR`` for per-job artifacts (profiling, LIKWID, mpiP, run1 copies).

    When ``artifact_dir`` is set in hpc_config (e.g. ``/scratch/%u/autotuner``), heavy I/O
    stays off GPFS/home; configs and reports remain under ``work_directory`` on the repo tree.
    """
    wd = Path(work_directory)
    art = ((system_config or {}).get("artifact_dir") or "").strip()
    if art:
        base = slurm_template_to_shell(art.rstrip("/"))
        return f"{base}/{wd.name}"
    try:
        return str(wd.expanduser().resolve())
    except OSError:
        return f"$SUBMIT_DIR/{wd.as_posix()}"


def experiment_artifact_dir_paths(
    work_directory: Union[str, Path],
    system_config: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Directories that may hold ``profiling/``, ``likwid_output/``, ``run1_*.out``, ``mpiP/``."""
    wd = Path(work_directory).expanduser()
    out: List[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = p.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(p)

    for root in experiment_path_roots(wd):
        add(root)
    art = ((system_config or {}).get("artifact_dir") or "").strip()
    if art:
        add(slurm_template_to_host_path(art.rstrip("/")) / wd.name)
    return out


def slurm_log_search_dirs(system_config: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Directories where ``#SBATCH --output=/scratch/%u/slurm-%j.out`` files are written."""
    dirs: List[Path] = []
    seen: set[str] = set()
    cfg = system_config or {}

    def add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = p.as_posix()
        if key not in seen:
            seen.add(key)
            dirs.append(p)

    sld = (cfg.get("slurm_log_dir") or "").strip()
    if sld:
        add(slurm_template_to_host_path(sld))
    scd = (cfg.get("submit_chdir") or "").strip()
    if scd:
        add(slurm_template_to_host_path(scd.rstrip("/")))
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if user:
        add(Path(f"/scratch/{user}"))
    return dirs


def find_slurm_log_for_job(
    experiment_dir: Union[str, Path],
    job_id: str,
    system_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Locate merged SLURM stdout/stderr for a job.

    Prefer ``slurm-<id>.out`` over ``.err`` (same ordering as ``gather_slurm_logs_by_job_id``).
    Checks experiment root, ``results/<job_id>/``, then tree-wide ``rglob`` so logs are found when
    SLURM paths differ from the archive layout (common with FAILED jobs or alternate cwd).
    """
    jid = str(job_id)
    roots = experiment_path_roots(Path(experiment_dir))

    for exp_dir in roots:
        for ext in (".out", ".err"):
            for p in (
                exp_dir / f"slurm-{jid}{ext}",
                exp_dir / "results" / jid / f"slurm-{jid}{ext}",
            ):
                if p.is_file():
                    return p

    for exp_dir in roots:
        for ext in (".out", ".err"):
            name = f"slurm-{jid}{ext}"
            for p in exp_dir.rglob(name):
                if p.is_file():
                    return p

    for exp_dir in roots:
        rd = exp_dir / "results" / jid
        if rd.is_dir():
            for pat in ("slurm-*.out", "slurm-*.err"):
                for p in sorted(rd.glob(pat)):
                    if p.is_file():
                        return p

    for log_dir in slurm_log_search_dirs(system_config):
        for ext in (".out", ".err"):
            p = log_dir / f"slurm-{jid}{ext}"
            if p.is_file():
                return p

    sacct_paths = slurm_log_paths_from_sacct(jid)
    sacct_paths.sort(key=lambda p: (0 if p.suffix.lower() == ".out" else 1, str(p)))
    for p in sacct_paths:
        if p.is_file():
            logger.info(
                "find_slurm_log: using sacct-reported path for job %s → %s",
                jid,
                p,
            )
            return p
    return None


def slurm_log_paths_from_sacct(job_id: str) -> List[Path]:
    """
    StdOut/StdErr paths from ``sacct`` when files are not under the experiment directory
    (site-specific Slurm output locations).
    """
    paths: List[Path] = []
    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "-n",
                "--parsable2",
                "--format=JobID,StdOut,StdErr",
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug("sacct unavailable for job %s: %s", job_id, e)
        return paths

    if proc.returncode != 0:
        logger.debug("sacct stderr for job %s: %s", job_id, proc.stderr or "")
        return paths

    seen: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid_part = parts[0].strip()
        if not jid_part.startswith(str(job_id)):
            continue
        for raw in (parts[1].strip(), parts[2].strip()):
            if not raw:
                continue
            expanded = raw.replace("%U", os.environ.get("USER", "") or "").replace(
                "%u", os.environ.get("USER", "") or ""
            )
            p = Path(expanded)
            try:
                rp = p.expanduser().resolve()
            except OSError:
                rp = p
            key = str(rp)
            if key in seen:
                continue
            seen.add(key)
            try:
                if rp.exists():
                    paths.append(rp)
            except OSError:
                pass
    return paths


def slurm_log_candidate_paths(
    job_id: str,
    system_config: Optional[Dict[str, Any]] = None,
    experiment_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Likely SLURM stdout/stderr paths for a job (for diagnostics when collection fails)."""
    jid = str(job_id)
    out: List[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)

    for d in slurm_log_search_dirs(system_config):
        for ext in (".err", ".out"):
            add(d / f"slurm-{jid}{ext}")
    if experiment_dir is not None:
        for root in experiment_path_roots(Path(experiment_dir)):
            for ext in (".err", ".out"):
                add(root / f"slurm-{jid}{ext}")
                add(root / "results" / jid / f"slurm-{jid}{ext}")
    for p in slurm_log_paths_from_sacct(jid):
        add(p)
    return out


def read_slurm_log_excerpt(
    job_id: str,
    system_config: Optional[Dict[str, Any]] = None,
    experiment_dir: Optional[Union[str, Path]] = None,
    max_lines: int = 50,
) -> tuple[Optional[Path], str]:
    """
    Read tail of SLURM stderr (preferred) or stdout for a failed job.

    Returns (path, excerpt_text). path is None if no file was found.
    """
    candidates = slurm_log_candidate_paths(job_id, system_config, experiment_dir)
    # Prefer .err, then .out
    ordered: List[Path] = []
    for p in candidates:
        if p.suffix.lower() == ".err":
            ordered.append(p)
    for p in candidates:
        if p.suffix.lower() == ".out":
            ordered.append(p)
    for path in ordered:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                continue
            text = path.read_text(errors="replace")
            lines = text.splitlines()
            excerpt = "\n".join(lines[-max_lines:]) if len(lines) > max_lines else text
            return path, excerpt.strip()
        except OSError:
            continue
    return None, ""


def log_sacct_job_diagnostic(job_id: str, prefix: str = "") -> None:
    """Emit State / ExitCode / Reason from sacct (helps diagnose FAILED jobs with missing slurm-*.out)."""
    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "-n",
                "--parsable2",
                "--format=JobID,State,ExitCode,DerivedExitCode,Reason",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning("%ssacct diagnostic failed for job %s: %s", prefix, job_id, e)
        return

    if proc.returncode != 0 or not proc.stdout.strip():
        logger.warning("%ssacct diagnostic empty for job %s: %s", prefix, job_id, proc.stderr or "")
        return

    # Slurm returns multiple rows (job id, .batch, .extern). Logging each row duplicated the
    # same WARNING twice — combine into one line for the terminal.
    rows = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if rows:
        merged = " | ".join(rows)
        logger.warning("%sjob %s — sacct: %s", prefix, job_id, merged)