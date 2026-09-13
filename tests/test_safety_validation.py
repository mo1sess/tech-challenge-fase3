from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from clinical_assistant.config import project_root
from clinical_assistant.validation.safety import validate_safety_stage


def _fixture(root: Path) -> Path:
    source = project_root()
    (root / "configs").mkdir()
    (root / "data" / "safety").mkdir(parents=True)
    shutil.copy(source / "configs" / "safety.yaml", root / "configs" / "safety.yaml")
    shutil.copy(source / "configs" / "agent.yaml", root / "configs" / "agent.yaml")
    shutil.copy(
        source / "data" / "safety" / "adversarial_cases.jsonl",
        root / "data" / "safety" / "adversarial_cases.jsonl",
    )
    report = root / "outputs" / "agent" / "stage8_validation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"stage": 8, "ok": True}), encoding="utf-8")
    return root


@pytest.mark.integration
def test_stage9_validator_confirms_guardrails_and_audit(tmp_path: Path) -> None:
    report = validate_safety_stage(_fixture(tmp_path), write_report=False)
    assert report["ok"] is True
    assert report["guardrails"]["cases"] == 8
    assert report["guardrails"]["blocked"] == 7
    assert report["guardrails"]["matched_expectations"] == 8
    assert report["graph"]["node_count"] == 11
    assert report["execution"]["gpu_used"] is False


@pytest.mark.unit
def test_stage9_validator_rejects_mutable_audit_configuration(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "configs" / "safety.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["audit"]["append_only"] = False
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    report = validate_safety_stage(root, write_report=False)
    assert report["ok"] is False
    assert "audit must remain append-only" in report["errors"]

