"""Safe, reproducible download and archive utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "tech-challenge-fase3/0.1 academic-project"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, force: bool = False) -> Path:
    """Download a URL atomically to destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting absolute paths and path traversal."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            member_path = (root / member.filename).resolve()
            if member_path != root and root not in member_path.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        zipped.extractall(root)


def flatten_single_directory(destination: Path) -> None:
    """Remove the single wrapper directory commonly used by GitHub archives."""

    entries = list(destination.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    wrapper = entries[0]
    for child in list(wrapper.iterdir()):
        shutil.move(str(child), destination / child.name)
    wrapper.rmdir()


def reset_directory(destination: Path) -> None:
    """Clear one dataset directory after its scope has been explicitly resolved."""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, human-readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

