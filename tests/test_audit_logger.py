from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_assistant.audit.audit_logger import AuditIntegrityError, JsonlAuditLogger


def _event(execution_id: str) -> dict:
    return {
        "execution_id": execution_id,
        "timestamp": "2026-09-13T00:00:00+00:00",
        "patient_id": "PAC004",
        "question": "Quais exames estão pendentes?",
        "tools_called": ["get_patient", "save_audit_log"],
        "retrieved_documents": [{"document_id": "ASM-001"}],
        "sources": ["Fonte sintética"],
        "model": "deterministic_evidence_preview",
        "response": "Resposta sintética.",
        "safety_result": {"status": "passed", "blocked": False},
        "human_validation_required": False,
        "human_validation_result": None,
    }


def test_audit_log_appends_events_with_valid_hash_chain(tmp_path: Path) -> None:
    logger = JsonlAuditLogger(tmp_path / "audit.jsonl")
    first = logger.record(_event("execution-1"))
    second = logger.record(_event("execution-2"))
    events = logger.read_events(limit=10)

    assert first["event_number"] == 1
    assert second["event_number"] == 2
    assert events[1]["previous_hash"] == events[0]["event_hash"]
    assert logger.verify()["last_hash"] == events[1]["event_hash"]


def test_audit_log_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = JsonlAuditLogger(path)
    logger.record(_event("execution-1"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response"] = "Conteúdo adulterado"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError, match="hash mismatch"):
        logger.verify()


def test_audit_log_rejects_missing_required_field(tmp_path: Path) -> None:
    event = _event("execution-1")
    event.pop("sources")
    with pytest.raises(ValueError, match="sources"):
        JsonlAuditLogger(tmp_path / "audit.jsonl").record(event)


def test_audit_log_applies_text_limits_and_has_no_delete_api(tmp_path: Path) -> None:
    logger = JsonlAuditLogger(
        tmp_path / "audit.jsonl", maximum_question_chars=10, maximum_response_chars=12
    )
    event = _event("execution-1")
    event["question"] = "q" * 30
    event["response"] = "r" * 30
    logger.record(event)
    saved = logger.read_events()[0]

    assert saved["question"] == "q" * 10
    assert saved["response"] == "r" * 12
    assert not hasattr(logger, "delete")


def test_audit_log_rejects_duplicate_execution_id(tmp_path: Path) -> None:
    logger = JsonlAuditLogger(tmp_path / "audit.jsonl")
    logger.record(_event("same-id"))
    with pytest.raises(ValueError, match="duplicate execution_id"):
        logger.record(_event("same-id"))
    assert logger.verify()["events"] == 1
