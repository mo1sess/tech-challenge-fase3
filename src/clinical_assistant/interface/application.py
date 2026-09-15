"""Framework-independent controller used by the Streamlit interface."""

from __future__ import annotations

import uuid
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.types import Command

from clinical_assistant.audit.audit_logger import JsonlAuditLogger
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.database.patient_repository import PatientRepository
from clinical_assistant.finetuning.remote import RemoteQwenResponseGenerator
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


def _answer_for_interface(response: str) -> str:
    """Remove blocks rendered separately by Streamlit without changing audit output."""

    answer = response.strip()
    sources_marker = "\n\nFontes recuperadas:\n"
    if sources_marker in answer:
        answer = answer.split(sources_marker, 1)[0].rstrip()
    disclaimer = (
        "Dados sintéticos para demonstração acadêmica. Não usar para diagnóstico, "
        "prescrição ou decisão clínica autônoma."
    )
    if answer.endswith(disclaimer):
        answer = answer[: -len(disclaimer)].rstrip()
    return answer


def present_workflow_result(
    result: dict[str, Any], *, thread_id: str
) -> dict[str, Any]:
    """Convert a LangGraph state into a stable view model for the UI."""

    review = _interrupt_payload(result)
    safety = dict(result.get("safety_result") or {})
    waiting = review is not None
    consistency = dict(result.get("response_consistency") or {})
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
        "answer": _answer_for_interface(str(result.get("final_response", ""))),
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
        "response_consistency": consistency,
        "structured_evidence_kind": str(consistency.get("evidence_kind", "")),
        "structured_evidence": list(consistency.get("evidence_records", [])),
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
    runtime_profile: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "local_preview",
            "generator": "deterministic_evidence_preview",
            "official_model_active": False,
        }
    )

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


def create_clinical_application(
    root: Path,
    *,
    generator: Any | None = None,
    runtime_profile: dict[str, Any] | None = None,
) -> ClinicalApplication:
    """Create the application with an explicit generator or the local preview."""

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
        graph=create_clinical_workflow(root, generator=generator),
        patient_repository=repository,
        audit_logger=audit_logger,
        runtime_profile=runtime_profile
        or {
            "mode": "local_preview",
            "generator": "deterministic_evidence_preview",
            "official_model_active": False,
            "model_id": "Qwen/Qwen3-8B",
        },
    )


def create_configured_clinical_application(
    root: Path,
    *,
    mode: str,
    endpoint_url: str = "",
    token: str = "",
) -> ClinicalApplication:
    """Select local preview or the verified official remote generator."""

    config = load_yaml(root / "configs" / "full_agent.yaml")
    local_mode = str(config["execution"]["local_mode"])
    official_mode = str(config["execution"]["official_mode"])
    selected = mode.strip() or local_mode
    if selected == local_mode:
        return create_clinical_application(root)
    if selected != official_mode:
        raise ValueError(
            f"Unsupported execution mode {selected!r}; use {local_mode!r} or {official_mode!r}"
        )

    manifest_path = resolve_project_path(config["model"]["adapter_manifest"], root=root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = str(
        manifest["artifact_sha256"][str(config["model"]["adapter_artifact"])]
    )
    generator = RemoteQwenResponseGenerator(
        endpoint_url=endpoint_url,
        token=token,
        timeout_seconds=float(config["execution"]["remote_timeout_seconds"]),
        expected_model_id=str(config["model"]["id"]),
        expected_revision=str(config["model"]["revision"]),
        expected_adapter_sha256=expected_hash,
    )
    health = generator.health()
    environment = dict(health.get("environment") or {})
    return create_clinical_application(
        root,
        generator=generator,
        runtime_profile={
            "mode": official_mode,
            "generator": generator.mode,
            "official_model_active": True,
            "model_id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "adapter_sha256": expected_hash,
            "endpoint_url": generator.endpoint_url,
            "environment": environment,
        },
    )
