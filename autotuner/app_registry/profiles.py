"""Application profile presets (from ``config/profiles.yaml``) and experiment-dir heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILES_YAML = _REPO_ROOT / "config" / "profiles.yaml"

# Fallback when PyYAML is missing or the file is absent
_DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "sparse_matrix": {
        "cpu_only": True,
        "enable_likwid": True,
        "description": "Sparse Matrix (irregular access, MPI-dominated)",
    },
    "hybrid_vec": {
        "cpu_only": False,
        "enable_likwid": True,
        "description": "Hybrid vector (MPI+OpenMP host + CUDA SAXPY, 1D)",
    },
    "minimd": {
        "cpu_only": False,
        "enable_likwid": True,
        "description": "Mantevo miniMD (MPI+OpenMP MD, particle-based)",
    },
    "lulesh": {
        "cpu_only": True,
        "enable_likwid": True,
        "description": "LULESH 2.x (MPI+OpenMP, structured mesh)",
    },
    "custom": {
        "cpu_only": None,
        "enable_likwid": None,
        "description": "Custom (user-configured)",
    },
}


def _yaml_to_profile_entry(key: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map ``profiles.yaml`` fields to CLI-oriented profile dict."""
    desc = raw.get("description") or raw.get("name") or key
    enable_locality = raw.get("enable_locality")
    if enable_locality is None:
        enable_locality = raw.get("enable_likwid")
    return {
        "cpu_only": raw.get("cpu_only"),
        "enable_likwid": enable_locality,
        "description": desc,
    }


def load_application_profiles() -> Dict[str, Dict[str, Any]]:
    """Load profiles from ``config/profiles.yaml``; fall back to built-in defaults."""
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed; using built-in APPLICATION_PROFILES defaults")
        return dict(_DEFAULT_PROFILES)

    if not _PROFILES_YAML.is_file():
        return dict(_DEFAULT_PROFILES)

    try:
        with open(_PROFILES_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not read %s: %s — using built-in profiles", _PROFILES_YAML, e)
        return dict(_DEFAULT_PROFILES)

    profiles: Dict[str, Dict[str, Any]] = {}
    for key, raw in data.items():
        if not isinstance(raw, dict) or key.startswith("_"):
            continue
        profiles[key] = _yaml_to_profile_entry(key, raw)
    if "custom" not in profiles:
        profiles["custom"] = _DEFAULT_PROFILES["custom"]
    return profiles or dict(_DEFAULT_PROFILES)


APPLICATION_PROFILES: Dict[str, Dict[str, Any]] = load_application_profiles()


def resolve_profile_key(application: str, application_profile: Optional[str]) -> str:
    """Choose which profile dict entry applies (``custom`` if unknown)."""
    key = (application_profile or "").strip() or "custom"
    if key == "custom" and application in APPLICATION_PROFILES:
        key = application
    return key if key in APPLICATION_PROFILES else "custom"


# ``master_tuner`` writes under ``data/experiments/{application}_{timestamp}``.
LATEST_EXPERIMENT_DIR_PREFIXES: tuple[str, ...] = (
    "sparse",
    "hybrid_vec",
    "lulesh",
    "minimd",
    "custom",
)


def yaml_profile_key_for_experiment_dir_name(exp_name_lower: str) -> str:
    """Map experiment folder name to a key in ``config/profiles.yaml``."""
    if "sparse" in exp_name_lower:
        return "sparse_matrix"
    if "hybrid_vec" in exp_name_lower:
        return "hybrid_vec"
    if "minimd" in exp_name_lower:
        return "minimd"
    if "lulesh" in exp_name_lower:
        return "lulesh"
    return "sparse_matrix"
