from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from clinical_assistant.database.builder import build_patient_database
from clinical_assistant.validation.database import validate_patient_database
from tests.patient_database_helpers import create_stage7_fixture


@pytest.mark.integration
def test_builder_creates_valid_database_and_manifest(tmp_path: Path) -> None:
    root = create_stage7_fixture(tmp_path)
    manifest = build_patient_database(root)
    report = validate_patient_database(root)

    assert manifest["table_counts"]["patients"] == 1
    assert manifest["safety"]["arbitrary_sql_allowed"] is False
    assert report["ok"] is True
    assert report["direct_identifiers_imported"] is False


@pytest.mark.unit
def test_validator_detects_database_tampering(tmp_path: Path) -> None:
    root = create_stage7_fixture(tmp_path)
    build_patient_database(root)
    database = root / "data" / "database" / "techcare.db"
    database.write_bytes(database.read_bytes() + b"tampered")

    report = validate_patient_database(root)

    assert report["ok"] is False
    assert "database SHA-256 does not match manifest" in report["errors"]


@pytest.mark.unit
def test_builder_rejects_pending_exam_without_academic_notice(tmp_path: Path) -> None:
    root = create_stage7_fixture(tmp_path)
    pending = root / "data" / "synthetic" / "hospital" / "pending_exams.jsonl"
    record = json.loads(pending.read_text(encoding="utf-8"))
    record["notice"] = ""
    pending.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks the synthetic notice"):
        build_patient_database(root)

    assert not (root / "data" / "database" / "techcare.db").exists()


@pytest.mark.unit
def test_builder_preserves_identical_source_rows(tmp_path: Path) -> None:
    root = create_stage7_fixture(tmp_path)
    observations = root / "data" / "processed" / "synthea_anonymized" / "observations.csv"
    with observations.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    with observations.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(rows[1])

    config_path = root / "configs" / "database.yaml"
    config = config_path.read_text(encoding="utf-8").replace("observations: 1", "observations: 2")
    config_path.write_text(config, encoding="utf-8")

    manifest = build_patient_database(root)

    assert manifest["table_counts"]["observations"] == 2
