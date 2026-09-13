"""Independent validation of stage 2 processed artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import utc_now, write_json
from clinical_assistant.preprocessing.anonymizer import SAFE_PATIENT_COLUMNS
from clinical_assistant.preprocessing.curation import content_hash


REQUIRED_EXAMPLE_FIELDS = {
    "category",
    "instruction",
    "input",
    "output",
    "source",
    "synthetic",
    "metadata",
    "content_hash",
}
PATIENT_REFERENCE_COLUMNS = {"PATIENT", "PATIENTID"}


def _validate_jsonl(path: Path) -> tuple[int, set[str], list[str]]:
    count = 0
    hashes: set[str] = set()
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                continue
            count += 1
            missing = REQUIRED_EXAMPLE_FIELDS - set(item)
            if missing:
                errors.append(f"{path.name}:{line_number}: missing {sorted(missing)}")
                continue
            fingerprint = str(item["content_hash"])
            if content_hash(item) != fingerprint:
                errors.append(f"{path.name}:{line_number}: content hash mismatch")
            if fingerprint in hashes:
                errors.append(f"{path.name}:{line_number}: duplicate content hash")
            hashes.add(fingerprint)
    return count, hashes, errors


def validate_training_splits(directory: Path) -> dict[str, Any]:
    """Check JSONL schema, fingerprints, duplicates, and cross-split leakage."""

    errors: list[str] = []
    counts: dict[str, int] = {}
    split_hashes: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"missing {path.name}")
            counts[split] = 0
            split_hashes[split] = set()
            continue
        count, hashes, split_errors = _validate_jsonl(path)
        counts[split] = count
        split_hashes[split] = hashes
        errors.extend(split_errors)

    overlaps = {
        "train_validation": len(split_hashes["train"] & split_hashes["validation"]),
        "train_test": len(split_hashes["train"] & split_hashes["test"]),
        "validation_test": len(split_hashes["validation"] & split_hashes["test"]),
    }
    if any(overlaps.values()):
        errors.append(f"cross-split leakage detected: {overlaps}")
    return {"ok": not errors, "counts": counts, "overlaps": overlaps, "errors": errors}


def validate_synthea_anonymization(directory: Path) -> dict[str, Any]:
    """Check safe patient columns and referential integrity across CSV tables."""

    errors: list[str] = []
    patients_path = directory / "patients.csv"
    if not patients_path.exists():
        return {"ok": False, "errors": ["missing patients.csv"]}

    with patients_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SAFE_PATIENT_COLUMNS:
            errors.append("patients.csv does not contain the approved anonymized schema")
        patient_ids = {row["Id"] for row in reader}
    if not patient_ids or any(not value.startswith("PAC") for value in patient_ids):
        errors.append("patient IDs are missing or do not use the PAC pseudonym format")

    table_counts: dict[str, int] = {}
    referenced_ids: set[str] = set()
    for path in sorted(directory.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = 0
            reference_fields = [
                field for field in (reader.fieldnames or []) if field.upper() in PATIENT_REFERENCE_COLUMNS
            ]
            for row in reader:
                rows += 1
                referenced_ids.update(row[field] for field in reference_fields if row[field])
            table_counts[path.stem] = rows
    unknown = sorted(referenced_ids - patient_ids)
    if unknown:
        errors.append(f"unknown anonymized patient references: {unknown[:5]}")
    return {
        "ok": not errors,
        "patients": len(patient_ids),
        "tables": len(table_counts),
        "table_counts": table_counts,
        "unknown_patient_references": len(unknown),
        "errors": errors,
    }


def validate_processed(root: Path) -> dict[str, Any]:
    """Validate all stage 2 outputs and write a consolidated report."""

    training = validate_training_splits(root / "data" / "processed" / "training")
    synthea = validate_synthea_anonymization(
        root / "data" / "processed" / "synthea_anonymized"
    )
    report = {
        "validated_at_utc": utc_now(),
        "stage": 2,
        "ok": bool(training["ok"] and synthea["ok"]),
        "training": training,
        "synthea": synthea,
    }
    write_json(root / "outputs" / "preprocessing_validation.json", report)
    return report

