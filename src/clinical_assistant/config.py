"""Configuration helpers shared by stage 0-1 commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root independently of the current directory."""

    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject unexpected top-level values."""

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_project_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a configured relative path without allowing escape from the root."""

    base = (root or project_root()).resolve()
    candidate = (base / value).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Path escapes project root: {value}")
    return candidate

