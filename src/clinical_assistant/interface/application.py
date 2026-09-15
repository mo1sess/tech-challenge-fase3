"""Framework-independent controller used by the Streamlit interface."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.types import Command

from clinical_assistant.audit.audit_logger import JsonlAuditLogger
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.database.patient_repository import PatientRepository
from clinical_assistant.graph.workflow import create_clinical_workflow


STATUS_LABELS = {
    "blocked": "Bloqueado pelos guardrails",
    "human_review_required": "Validação médica necessária",
    "safe": "Consulta informativa",
}


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return dict(value) if isinstance(value, dict) else {}


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def present_workflow_result(
    result: dict[str, Any], *, thread_id: str
) -> dict[str, Any]:
    """Convert a LangGraph state into a stable view model for the UI."""

    review = _interrupt_payload(result)
    safety = dict(result.get("safety_result") or {})
    waiting = review is not None
    status = "awaiting_human_review" if waiting else "completed"
    if safety.get("blocked"):
        safety_label = STATUS_LABELS["blocked"]
    elif result.get("requires_human_validation"):
        safety_label = STATUS_LABELS["human_review_required"]
    else:
        safety_label = STATUS_LABELS["safe"]

    return {
        "status": status,
        "thread_id": thread_id,
        "execution_id": str(result.get("execution_id", thread_id)),
        "patient_id": str(result.get("patient_id", "")),
        "question": str(result.get("question", "")),
        "answer": str(result.get("final_response", "")),
        "draft": str((review or {}).get("draft", "")),
        "sources": _unique_strings(list(result.get("citations", []))),
        "pending_exams": list(result.get("pending_exams", [])),
        "retrieved_documents": list(result.get("retrieved_documents", [])),
        "safety": safety,
        "safety_label": safety_label,
        "requires_human_validation": bool(
            result.get("requires_human_validation", False)
        ),
        "human_validation": result.get("human_validation"),
        "review_request": review,
        "generator_mode": str(result.get("generator_mode", "not_executed")),
        "tools_called": list(result.get("selected_tools", [])),
        "limitations": _unique_strings(list(result.get("limitations", []))),
        "audit_result": dict(result.get("audit_result") or {}),
        "trace": list(result.get("trace", [])),
    }


@dataclass
class ClinicalApplication:
    """Coordinate patient selection, graph execution, review, and audit reads."""

    graph: Any
    patient_repository: Any
    audit_logger: Any

    def list_patient_ids(self) -> list[str]:
        values = self.patient_repository.list_patient_ids()
        return sorted(_unique_strings(list(values)))

    def submit_question(
        self, patient_id: str, question: str, *, thread_id: str | None = None
    ) -> dict[str, Any]:
        run_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": run_id}}
        result = self.graph.invoke(
            {
                "patient_id": patient_id,
                "question": question,
                "execution_id": run_id,
                "trace": [],
            },
            config=config,
        )
        return present_workflow_result(result, thread_id=run_id)

    def resume_review(
        self,
        thread_id: str,
        *,
        approved: bool,
        notes: str = "",
    ) -> dict[str, Any]:
        if not thread_id.strip():
            raise ValueError("thread_id cannot be empty")
        result = self.graph.invoke(
            Command(
                resume={
                    "approved": bool(approved),
                    "reviewer": "streamlit_demo_user",
                    "notes": " ".join(notes.split())[:500],
                }
            ),
            config={"configurable": {"thread_id": thread_id}},
        )
        return present_workflow_result(result, thread_id=thread_id)

    def recent_audit_events(self, *, limit: int = 20) -> dict[str, Any]:
        integrity = self.audit_logger.verify()
        events = self.audit_logger.read_events(limit=limit)
        return {"integrity": integrity, "events": events}


def create_clinical_application(root: Path) -> ClinicalApplication:
    """Create the local CPU application without loading the official Qwen3-8B."""

    database_config = load_yaml(root / "configs" / "database.yaml")["database"]
    safety_config = load_yaml(root / "configs" / "safety.yaml")["audit"]
    repository = PatientRepository(
        resolve_project_path(database_config["path"], root=root),
        maximum_result_limit=int(database_config["maximum_result_limit"]),
    )
    audit_logger = JsonlAuditLogger(
        resolve_project_path(safety_config["path"], root=root),
        maximum_question_chars=int(safety_config["maximum_question_chars"]),
        maximum_response_chars=int(safety_config["maximum_response_chars"]),
        maximum_events_to_read=int(safety_config["maximum_events_to_read"]),
    )
    return ClinicalApplication(
        graph=create_clinical_workflow(root),
        patient_repository=repository,
        audit_logger=audit_logger,
    )
