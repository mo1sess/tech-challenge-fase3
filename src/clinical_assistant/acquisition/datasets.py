"""Acquisition and counting logic for the three public datasets."""

from __future__ import annotations

import csv
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

from clinical_assistant.acquisition.common import (
    download_file,
    flatten_single_directory,
    reset_directory,
    safe_extract_zip,
    sha256_file,
    utc_now,
    write_json,
)


def count_medquad(destination: Path) -> dict[str, int]:
    """Count parseable XML documents and QAPair elements."""

    xml_files = sorted(destination.rglob("*.xml"))
    pairs = 0
    for path in xml_files:
        root = ET.parse(path).getroot()
        pairs += sum(1 for _ in root.iter() if _.tag.rsplit("}", 1)[-1] == "QAPair")
    return {"xml_files": len(xml_files), "qa_pairs": pairs}


def count_pubmedqa(destination: Path) -> dict[str, int]:
    """Count official labeled PubMedQA examples."""

    matches = list(destination.rglob("ori_pqal.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one ori_pqal.json, found {len(matches)}")
    with matches[0].open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("PubMedQA PQA-L must be a JSON object keyed by PMID")
    return {"pqa_l_records": len(payload)}


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def count_synthea(destination: Path) -> dict[str, int]:
    """Count Synthea CSV entities, with patients as the primary record count."""

    csv_files = sorted(destination.rglob("*.csv"))
    counts = {path.stem.lower(): _csv_rows(path) for path in csv_files}
    counts["csv_files"] = len(csv_files)
    return counts


COUNTERS: dict[str, Callable[[Path], dict[str, int]]] = {
    "medquad": count_medquad,
    "pubmedqa": count_pubmedqa,
    "synthea": count_synthea,
}


def acquire_archive(
    name: str,
    config: dict[str, Any],
    *,
    project_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Download, safely extract, count, and record one configured dataset."""

    if name not in COUNTERS:
        raise ValueError(f"Unsupported downloadable dataset: {name}")
    destination = (project_root / config["destination"]).resolve()
    if project_root.resolve() not in destination.parents:
        raise ValueError(f"Dataset destination escapes project root: {destination}")

    manifest_path = project_root / "data" / "raw" / "_manifests" / f"{name}.json"
    has_payload = destination.exists() and any(
        path.name != ".gitkeep" for path in destination.iterdir()
    )
    if not force and has_payload and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if isinstance(existing, dict):
            return existing

    with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=project_root / "data") as temp:
        archive = Path(temp) / f"{name}.zip"
        download_file(str(config["url"]), archive, force=True)
        archive_hash = sha256_file(archive)
        reset_directory(destination)
        safe_extract_zip(archive, destination)
        flatten_single_directory(destination)

    counts = COUNTERS[name](destination)
    manifest = {
        "dataset": name,
        "kind": config["kind"],
        "source_url": config["url"],
        "revision": config["revision"],
        "downloaded_at_utc": utc_now(),
        "archive_sha256": archive_hash,
        "counts": counts,
    }
    write_json(manifest_path, manifest)
    return manifest
