from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.finetuning.dataset import (
    prepare_finetuning_dataset,
    split_records_by_category,
    validate_finetuning_dataset,
)


def _temporary_root(tmp_path: Path) -> Path:
    source_root = project_root()
    (tmp_path / "configs").mkdir()
    shutil.copy(source_root / "configs" / "training.yaml", tmp_path / "configs")
    synthetic = tmp_path / "data" / "synthetic" / "hospital"
    synthetic.mkdir(parents=True)
    for path in (source_root / "data" / "synthetic" / "hospital").glob("*.jsonl"):
        shutil.copy(path, synthetic)
    evaluation = tmp_path / "data" / "evaluation"
    evaluation.mkdir(parents=True)
    for path in (source_root / "data" / "evaluation").glob("*.jsonl"):
        shutil.copy(path, evaluation)
    return tmp_path


@pytest.mark.integration
def test_preparation_creates_stratified_leakage_free_splits(tmp_path: Path) -> None:
    root = _temporary_root(tmp_path)
    manifest = prepare_finetuning_dataset(root)
    report = validate_finetuning_dataset(root)

    assert manifest["source_records"] == 94
    assert manifest["leakage_check"]["ok"] is True
    assert report["ok"] is True
    assert report["counts"] == {"test": 8, "train": 78, "validation": 8}
    assert report["unique_content_hashes"] == 94
    expected_categories = {
        "faq", "prescription_behavior", "procedure", "protocol", "report", "safety"
    }
    assert all(set(counts) == expected_categories for counts in report["categories"].values())


@pytest.mark.integration
def test_preparation_is_deterministic(tmp_path: Path) -> None:
    root = _temporary_root(tmp_path)
    first = prepare_finetuning_dataset(root)
    second = prepare_finetuning_dataset(root)
    assert first["files"] == second["files"]


@pytest.mark.unit
def test_split_rejects_invalid_percentages() -> None:
    with pytest.raises(ValueError, match="sum to 100"):
        split_records_by_category(
            [], train_percent=70, validation_percent=10, test_percent=10, seed=42
        )


@pytest.mark.unit
def test_validator_detects_changed_split(tmp_path: Path) -> None:
    root = _temporary_root(tmp_path)
    prepare_finetuning_dataset(root)
    path = root / "data" / "processed" / "finetuning" / "train.jsonl"
    records = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(records[0])
    record["completion"][0]["content"] = "conteúdo alterado"
    records[0] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    report = validate_finetuning_dataset(root)
    assert report["ok"] is False
    assert "SHA-256 mismatch: train.jsonl" in report["errors"]
