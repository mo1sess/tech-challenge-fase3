from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_assistant.preprocessing.dataset_builder import iter_medquad, iter_pubmedqa


@pytest.mark.unit
def test_iter_medquad_preserves_provenance(tmp_path: Path) -> None:
    (tmp_path / "sample.xml").write_text(
        """<Document source="NIH" url="https://example.test/doc">
        <Focus>Asthma</Focus><QAPairs><QAPair><Question qid="q1" qtype="treatment">
        How is asthma monitored?</Question><Answer>With clinical follow-up.</Answer>
        </QAPair></QAPairs></Document>""",
        encoding="utf-8",
    )
    item = next(iter_medquad(tmp_path))
    assert item["instruction"] == "How is asthma monitored?"
    assert item["output"] == "With clinical follow-up."
    assert item["metadata"]["question_id"] == "q1"
    assert item["metadata"]["document_url"] == "https://example.test/doc"


@pytest.mark.unit
def test_iter_pubmedqa_builds_evidence_input(tmp_path: Path) -> None:
    path = tmp_path / "ori_pqal.json"
    path.write_text(
        json.dumps(
            {
                "123": {
                    "QUESTION": "Does intervention X help?",
                    "CONTEXTS": ["First result.", "Second result."],
                    "MESHES": ["Asthma"],
                    "YEAR": "2020",
                    "final_decision": "yes",
                    "LONG_ANSWER": "The results support the intervention.",
                }
            }
        ),
        encoding="utf-8",
    )
    item = next(iter_pubmedqa(path))
    assert "Evidence 1: First result." in item["input"]
    assert item["output"].startswith("Decision: yes.")
    assert item["metadata"]["pmid"] == "123"

