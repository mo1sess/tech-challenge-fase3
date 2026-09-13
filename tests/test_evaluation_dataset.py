from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.evaluation.dataset import (
    load_evaluation_cases,
    normalize_for_comparison,
    validate_evaluation_dataset,
)


def _copy_evaluation_fixture(tmp_path: Path) -> Path:
    root = project_root()
    (tmp_path / "configs").mkdir()
    shutil.copy(root / "configs" / "baseline.yaml", tmp_path / "configs")
    destination = tmp_path / "data" / "evaluation"
    destination.mkdir(parents=True)
    for path in (root / "data" / "evaluation").glob("*.jsonl"):
        shutil.copy(path, destination)
    synthetic = tmp_path / "data" / "synthetic" / "hospital"
    synthetic.mkdir(parents=True)
    for path in (root / "data" / "synthetic" / "hospital").glob("*.jsonl"):
        shutil.copy(path, synthetic)
    return tmp_path


@pytest.mark.unit
def test_normalization_is_accent_and_whitespace_insensitive() -> None:
    assert normalize_for_comparison("  Validação   Médica ") == "validacao medica"


@pytest.mark.integration
def test_evaluation_dataset_is_complete_and_leakage_free(tmp_path: Path) -> None:
    root = _copy_evaluation_fixture(tmp_path)
    report = validate_evaluation_dataset(root)
    assert report["ok"] is True
    assert report["total_cases"] == 24
    assert report["counts"] == {"clinical": 6, "patient": 6, "protocol": 6, "safety": 6}
    assert report["exact_leakage_count"] == 0


@pytest.mark.unit
def test_loader_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_evaluation_cases(tmp_path, ["bad.jsonl"])


@pytest.mark.integration
def test_validator_detects_duplicate_question(tmp_path: Path) -> None:
    root = _copy_evaluation_fixture(tmp_path)
    path = root / "data" / "evaluation" / "clinical_questions.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["question"] = records[0]["question"]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    report = validate_evaluation_dataset(root)
    assert report["ok"] is False
    assert any("duplicate question" in error for error in report["errors"])


@pytest.mark.unit
def test_validator_ignores_blank_lines_in_training_jsonl(tmp_path: Path) -> None:
    root = _copy_evaluation_fixture(tmp_path)
    path = root / "data" / "synthetic" / "hospital" / "safety.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = validate_evaluation_dataset(root)

    assert report["ok"] is True
