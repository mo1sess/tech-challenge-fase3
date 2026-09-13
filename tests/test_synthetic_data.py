from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.synthetic.catalog import NOTICE, build_catalog
from clinical_assistant.synthetic.generator import generate_synthetic_hospital
from clinical_assistant.validation.synthetic import validate_synthetic_hospital


EXPECTED_COUNTS = {
    "protocols": 15,
    "medical_faq": 20,
    "reports": 12,
    "prescriptions": 12,
    "procedures": 15,
    "safety": 20,
}


def _temporary_root(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    shutil.copy(project_root() / "configs" / "synthetic_data.yaml", config_dir)
    return tmp_path


@pytest.mark.unit
def test_catalog_has_expected_counts_and_labels() -> None:
    catalog = build_catalog()
    assert {name: len(records) for name, records in catalog.items()} == EXPECTED_COUNTS
    records = [record for values in catalog.values() for record in values]
    assert len({record["id"] for record in records}) == 94
    assert len({record["content_hash"] for record in records}) == 94
    assert all(record["synthetic"] is True for record in records)
    assert all(record["notice"] == NOTICE for record in records)


@pytest.mark.integration
def test_generated_catalog_passes_independent_validation(tmp_path: Path) -> None:
    root = _temporary_root(tmp_path)
    manifest = generate_synthetic_hospital(root)
    report = validate_synthetic_hospital(root)
    assert manifest["total_records"] == 94
    assert report["ok"] is True
    assert report["unique_ids"] == 94
    assert report["required_protocols_found"] == [
        "ASM-001",
        "ASM-002",
        "ASM-003",
        "ASM-004",
        "ASM-005",
    ]


@pytest.mark.unit
def test_validator_rejects_missing_synthetic_notice(tmp_path: Path) -> None:
    root = _temporary_root(tmp_path)
    generate_synthetic_hospital(root)
    path = root / "data" / "synthetic" / "hospital" / "medical_faq.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["notice"] = ""
    lines[0] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_synthetic_hospital(root)
    assert report["ok"] is False
    assert any("missing synthetic academic notice" in error for error in report["errors"])


@pytest.mark.unit
def test_validator_rejects_autonomous_clinical_instruction(tmp_path: Path) -> None:
    root = _temporary_root(tmp_path)
    generate_synthetic_hospital(root)
    path = root / "data" / "synthetic" / "hospital" / "safety.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["output"] = "Prescrevo este medicamento sem revisão."
    lines[0] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_synthetic_hospital(root)
    assert report["ok"] is False
    assert any("autonomous clinical instruction" in error for error in report["errors"])
