from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from clinical_assistant.config import project_root
from clinical_assistant.validation.agent import validate_agent_stage


def _fixture(root: Path) -> Path:
    (root / "configs").mkdir()
    shutil.copy(project_root() / "configs" / "agent.yaml", root / "configs" / "agent.yaml")
    for stage, relative in (
        (6, Path("outputs/rag/index_manifest.json")),
        (7, Path("outputs/database/database_manifest.json")),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stage": stage, "complete": True}
        if stage == 6:
            payload["source"] = {"logical_documents": 5}
        else:
            payload["table_counts"] = {"patients": 108}
        path.write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.mark.integration
def test_stage8_validator_confirms_dependencies_and_prerequisites(tmp_path: Path) -> None:
    report = validate_agent_stage(_fixture(tmp_path), write_report=False)
    assert report["ok"] is True
    assert report["graph"]["node_count"] == 9
    assert report["prerequisites"] == {
        "rag_stage": 6,
        "rag_documents": 5,
        "database_stage": 7,
        "database_patients": 108,
    }
    assert report["execution"]["gpu_used"] is False


@pytest.mark.unit
def test_stage8_validator_rejects_unsafe_configuration(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "configs" / "agent.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["safety"]["allow_arbitrary_sql"] = True
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    report = validate_agent_stage(root, write_report=False)

    assert report["ok"] is False
    assert "arbitrary SQL must remain disabled" in report["errors"]
