"""Controlled import of user-provided protocol documents."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json


def import_protocols(
    inbox: Path,
    destination: Path,
    *,
    allowed_extensions: set[str],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Copy allowed local documents without interpreting them as clinical truth."""

    inbox.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    imported: list[dict[str, Any]] = []
    rejected: list[str] = []
    for source in sorted(path for path in inbox.rglob("*") if path.is_file()):
        if source.name == "README.md":
            continue
        if source.suffix.lower() not in allowed_extensions:
            rejected.append(source.relative_to(inbox).as_posix())
            continue
        relative = source.relative_to(inbox)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        imported.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
                "classification": "local_protocol_unreviewed",
            }
        )
    result = {
        "dataset": "protocols",
        "kind": "local_or_public",
        "imported_at_utc": utc_now(),
        "counts": {"imported_files": len(imported), "rejected_files": len(rejected)},
        "files": imported,
        "rejected": rejected,
    }
    if manifest_path:
        write_json(manifest_path, result)
    return result

