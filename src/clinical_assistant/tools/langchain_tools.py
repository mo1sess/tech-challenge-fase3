"""LangChain tools exposing only bounded, read-only project operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from clinical_assistant.rag.models import RetrievalHit
from clinical_assistant.tools.patient_tools import PatientTools


class ProtocolSearch(Protocol):
    def retrieve(self, query: str) -> list[RetrievalHit]: ...


class AuditSink(Protocol):
    def record(self, event: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LangChainToolbox:
    tools: tuple[BaseTool, ...]

    def by_name(self) -> dict[str, BaseTool]:
        return {tool.name: tool for tool in self.tools}


def create_langchain_toolbox(
    patient_tools: PatientTools,
    protocol_retriever: ProtocolSearch,
    *,
    guideline_limitation: str,
    audit_logger: AuditSink | None = None,
) -> LangChainToolbox:
    """Create named tools without a generic SQL or filesystem escape hatch."""

    def get_patient(patient_id: str) -> dict[str, Any]:
        """Return the minimal pseudonymized patient record."""

        return patient_tools.get_patient(patient_id)

    def get_patient_conditions(patient_id: str) -> dict[str, Any]:
        """Return bounded condition records for one pseudonymized patient."""

        return patient_tools.get_patient_conditions(patient_id)

    def get_patient_medications(patient_id: str) -> dict[str, Any]:
        """Return bounded medication records for one pseudonymized patient."""

        return patient_tools.get_patient_medications(patient_id)

    def get_pending_exams(patient_id: str) -> dict[str, Any]:
        """Return only explicitly registered synthetic pending exams."""

        return patient_tools.get_pending_exams(patient_id)

    def get_patient_observations(patient_id: str) -> dict[str, Any]:
        """Return bounded recent observations for one pseudonymized patient."""

        return patient_tools.get_patient_observations(patient_id)

    def search_internal_protocol(query: str) -> dict[str, Any]:
        """Search the synthetic internal protocol corpus and include citations."""

        hits = protocol_retriever.retrieve(query)
        return {
            "ok": True,
            "query": query,
            "documents": [hit.to_dict() for hit in hits],
            "citations": [hit.citation for hit in hits],
            "synthetic": True,
            "warning": "Protocolos internos sintéticos; validação médica necessária.",
        }

    def search_clinical_guideline(query: str) -> dict[str, Any]:
        """Report that no reviewed official clinical guideline is currently loaded."""

        return {
            "ok": False,
            "query": query,
            "documents": [],
            "citations": [],
            "limitation": guideline_limitation,
        }

    def save_audit_log(event: dict[str, Any]) -> dict[str, Any]:
        """Append one validated workflow event to the integrity-protected audit log."""

        if audit_logger is None:
            raise RuntimeError("audit logger is not configured")
        return audit_logger.record(event)

    functions = (
        get_patient,
        get_patient_conditions,
        get_patient_medications,
        get_pending_exams,
        get_patient_observations,
        search_internal_protocol,
        search_clinical_guideline,
    )
    tools = [StructuredTool.from_function(function) for function in functions]
    if audit_logger is not None:
        tools.append(StructuredTool.from_function(save_audit_log))
    return LangChainToolbox(tuple(tools))
