"""Independent integrity and safety checks for stage 3 synthetic datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.preprocessing.anonymizer import CPF_PATTERN, EMAIL_PATTERN
from clinical_assistant.preprocessing.curation import content_hash
from clinical_assistant.synthetic.generator import FILE_TO_CATEGORY


REQUIRED_FIELDS = {
    "id",
    "category",
    "instruction",
    "input",
    "output",
    "source",
    "synthetic",
    "notice",
    "metadata",
    "content_hash",
}
REQUIRED_METADATA = {
    "document_id",
    "document_type",
    "version",
    "section",
    "requires_human_validation",
    "risk_level",
}
AUTONOMOUS_CLINICAL_OUTPUT = re.compile(
    r"\b(?:prescrevo|inicie|suspenda|aumente\s+a\s+dose|reduza\s+a\s+dose)\b",
    re.IGNORECASE,
)
PATIENT_CODE = re.compile(r"\bPAC\d+\b")


def _validate_record(
    record: dict[str, Any],
    *,
    expected_category: str,
    source: str,
    notice: str,
) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        return [f"missing fields {sorted(missing)}"]
    if record["category"] != expected_category:
        errors.append(f"unexpected category {record['category']!r}")
    if record["source"] != source:
        errors.append("unexpected or non-fictional source")
    if record["notice"] != notice:
        errors.append("missing synthetic academic notice")
    if record["synthetic"] is not True:
        errors.append("synthetic flag must be true")
    if content_hash(record) != record["content_hash"]:
        errors.append("content hash mismatch")

    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return errors
    metadata_missing = REQUIRED_METADATA - set(metadata)
    if metadata_missing:
        errors.append(f"missing metadata {sorted(metadata_missing)}")
    output = str(record["output"])
    if metadata.get("requires_human_validation") is True:
        if "validação médica necessária" not in output.casefold():
            errors.append("human validation flag lacks an explicit output warning")
    if AUTONOMOUS_CLINICAL_OUTPUT.search(output):
        errors.append("output contains an autonomous clinical instruction")

    combined = "\n".join(str(record[field]) for field in ("instruction", "input", "output"))
    if EMAIL_PATTERN.search(combined) or CPF_PATTERN.search(combined):
        errors.append("direct personal identifier pattern found")
    invalid_patient_codes = {
        match.group(0) for match in PATIENT_CODE.finditer(combined) if len(match.group(0)) != 6
    }
    if invalid_patient_codes:
        errors.append(f"invalid patient pseudonyms {sorted(invalid_patient_codes)}")
    return errors


def validate_synthetic_hospital(root: Path) -> dict[str, Any]:
    """Validate counts, provenance, safety properties, hashes, and manifest."""

    config = load_yaml(root / "configs" / "synthetic_data.yaml")
    output_dir = resolve_project_path(config["output_directory"], root=root)
    expected_counts = {
        str(key): int(value) for key, value in config["expected_counts"].items()
    }
    errors: list[str] = []
    counts: dict[str, int] = {}
    identifiers: set[str] = set()
    hashes: set[str] = set()
    protocol_ids: set[str] = set()

    for stem, category in FILE_TO_CATEGORY.items():
        path = output_dir / f"{stem}.jsonl"
        if not path.exists():
            errors.append(f"missing {path.name}")
            counts[category] = 0
            continue
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                    continue
                count += 1
                if not isinstance(record, dict):
                    errors.append(f"{path.name}:{line_number}: record must be an object")
                    continue
                for error in _validate_record(
                    record,
                    expected_category=category,
                    source=str(config["source"]),
                    notice=str(config["notice"]),
                ):
                    errors.append(f"{path.name}:{line_number}: {error}")
                identifier = str(record.get("id", ""))
                fingerprint = str(record.get("content_hash", ""))
                if identifier in identifiers:
                    errors.append(f"{path.name}:{line_number}: duplicate id {identifier}")
                if fingerprint in hashes:
                    errors.append(f"{path.name}:{line_number}: duplicate content hash")
                identifiers.add(identifier)
                hashes.add(fingerprint)
                metadata = record.get("metadata", {})
                if category == "protocol" and isinstance(metadata, dict):
                    protocol_ids.add(str(metadata.get("document_id", "")))
        counts[category] = count
        if count != expected_counts[category]:
            errors.append(
                f"{path.name}: unexpected count {count}, expected {expected_counts[category]}"
            )

    required_protocols = set(map(str, config["required_protocols"]))
    missing_protocols = sorted(required_protocols - protocol_ids)
    if missing_protocols:
        errors.append(f"missing required protocols {missing_protocols}")

    manifest_path = resolve_project_path(config["manifest"], root=root)
    manifest_ok = False
    if not manifest_path.exists():
        errors.append("missing synthetic_manifest.json")
    else:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest_ok = True
        if manifest.get("total_records") != sum(counts.values()):
            errors.append("manifest total count mismatch")
            manifest_ok = False
        for stem in FILE_TO_CATEGORY:
            path = output_dir / f"{stem}.jsonl"
            expected_file = manifest.get("files", {}).get(path.name, {})
            if path.exists() and expected_file.get("sha256") != sha256_file(path):
                errors.append(f"manifest hash mismatch for {path.name}")
                manifest_ok = False

    report = {
        "validated_at_utc": utc_now(),
        "stage": 3,
        "ok": not errors,
        "total_records": sum(counts.values()),
        "counts": counts,
        "unique_ids": len(identifiers),
        "unique_content_hashes": len(hashes),
        "required_protocols_found": sorted(protocol_ids & required_protocols),
        "manifest_ok": manifest_ok,
        "errors": errors,
    }
    write_json(root / "outputs" / "synthetic_data_validation.json", report)
    return report
