from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from clinical_assistant.acquisition.datasets import (
    count_medquad,
    count_pubmedqa,
    count_synthea,
)


@pytest.mark.unit
def test_count_medquad(tmp_path: Path) -> None:
    (tmp_path / "sample.xml").write_text(
        "<Document><QAPairs><QAPair/><QAPair/></QAPairs></Document>", encoding="utf-8"
    )
    assert count_medquad(tmp_path) == {"xml_files": 1, "qa_pairs": 2}


@pytest.mark.unit
def test_count_pubmedqa(tmp_path: Path) -> None:
    (tmp_path / "ori_pqal.json").write_text(
        json.dumps({"1": {"QUESTION": "A?"}, "2": {"QUESTION": "B?"}}), encoding="utf-8"
    )
    assert count_pubmedqa(tmp_path) == {"pqa_l_records": 2}


@pytest.mark.unit
def test_count_synthea(tmp_path: Path) -> None:
    with (tmp_path / "patients.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Id", "BIRTHDATE"])
        writer.writeheader()
        writer.writerows([{"Id": "1", "BIRTHDATE": "2000-01-01"}, {"Id": "2", "BIRTHDATE": "2001-01-01"}])
    assert count_synthea(tmp_path) == {"patients": 2, "csv_files": 1}

