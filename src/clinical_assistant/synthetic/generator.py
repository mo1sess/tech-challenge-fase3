"""Generate deterministic synthetic internal datasets and their manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.synthetic.catalog import build_catalog


FILE_TO_CATEGORY = {
    "protocols": "protocol",
    "medical_faq": "faq",
    "reports": "report",
    "prescriptions": "prescription_behavior",
    "procedures": "procedure",
    "safety": "safety",
}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def generate_synthetic_hospital(root: Path) -> dict[str, Any]:
    """Create all stage 3 artifacts from the reviewed in-code catalog."""

    config = load_yaml(root / "configs" / "synthetic_data.yaml")
    output_dir = resolve_project_path(config["output_directory"], root=root)
    catalog = build_catalog()
    expected = {str(key): int(value) for key, value in config["expected_counts"].items()}

    counts: dict[str, int] = {}
    files: dict[str, dict[str, Any]] = {}
    all_hashes: set[str] = set()
    for stem, records in catalog.items():
        category = FILE_TO_CATEGORY[stem]
        if len(records) != expected[category]:
            raise ValueError(
                f"Unexpected count for {stem}: {len(records)} != {expected[category]}"
            )
        for record in records:
            fingerprint = str(record["content_hash"])
            if fingerprint in all_hashes:
                raise ValueError(f"Duplicate synthetic content: {fingerprint}")
            all_hashes.add(fingerprint)
        path = output_dir / f"{stem}.jsonl"
        _write_jsonl(path, records)
        counts[category] = len(records)
        files[path.name] = {
            "category": category,
            "records": len(records),
            "sha256": sha256_file(path),
        }

    document_ids = sorted(
        {
            str(record["metadata"]["document_id"])
            for records in catalog.values()
            for record in records
        }
    )
    manifest = {
        "schema_version": 1,
        "stage": 3,
        "generated_at_utc": utc_now(),
        "seed": int(config["seed"]),
        "hospital": str(config["hospital_name"]),
        "source": str(config["source"]),
        "notice": str(config["notice"]),
        "synthetic": True,
        "total_records": sum(counts.values()),
        "counts": counts,
        "files": files,
        "document_ids": document_ids,
        "limitations": [
            "Fictional academic material; not a real hospital policy.",
            "No medication, dose, diagnosis, or clinical recommendation was authored.",
            "No official asthma guideline was available for clinical grounding in this stage.",
            "Content requires professional and ethics review before any non-academic use.",
        ],
    }
    write_json(resolve_project_path(config["manifest"], root=root), manifest)
    return manifest
