from __future__ import annotations

import csv
from pathlib import Path

import pytest

from clinical_assistant.preprocessing.curation import curate_examples
from clinical_assistant.preprocessing.dataset_builder import split_examples, write_jsonl
from clinical_assistant.validation.preprocessing import (
    validate_synthea_anonymization,
    validate_training_splits,
)


def _curated_examples() -> list[dict[str, object]]:
    raw = [
        {
            "category": "medical_qa",
            "instruction": f"Clinical question {index}",
            "input": "Context",
            "output": f"Clinical answer {index}",
            "source": "fixture",
            "synthetic": False,
            "metadata": {},
        }
        for index in range(50)
    ]
    return curate_examples(raw)[0]


@pytest.mark.unit
def test_training_validation_accepts_clean_partitions(tmp_path: Path) -> None:
    partitions = split_examples(
        _curated_examples(), {"train": 80, "validation": 10, "test": 10}
    )
    for split, records in partitions.items():
        write_jsonl(tmp_path / f"{split}.jsonl", records)
    result = validate_training_splits(tmp_path)
    assert result["ok"] is True
    assert sum(result["counts"].values()) == 50
    assert all(value == 0 for value in result["overlaps"].values())


@pytest.mark.unit
def test_training_validation_detects_cross_split_leakage(tmp_path: Path) -> None:
    example = _curated_examples()[0]
    write_jsonl(tmp_path / "train.jsonl", [example])
    write_jsonl(tmp_path / "validation.jsonl", [example])
    write_jsonl(tmp_path / "test.jsonl", [])
    result = validate_training_splits(tmp_path)
    assert result["ok"] is False
    assert result["overlaps"]["train_validation"] == 1


@pytest.mark.unit
def test_synthea_validation_detects_unknown_reference(tmp_path: Path) -> None:
    with (tmp_path / "patients.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "Id", "BIRTH_YEAR", "DEATH_YEAR", "MARITAL", "RACE", "ETHNICITY",
            "GENDER", "STATE", "COUNTY", "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE", "INCOME",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"Id": "PAC001"})
    with (tmp_path / "conditions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["PATIENT", "DESCRIPTION"])
        writer.writeheader()
        writer.writerow({"PATIENT": "PAC999", "DESCRIPTION": "Asthma"})
    result = validate_synthea_anonymization(tmp_path)
    assert result["ok"] is False
    assert result["unknown_patient_references"] == 1

