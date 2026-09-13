from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical_assistant.chains.context_chain import ContextBuilderChain
from clinical_assistant.graph.workflow import ClinicalWorkflow, DeterministicEvidencePreview
from clinical_assistant.rag.models import RetrievalHit
from clinical_assistant.safety.guardrails import SafetyGuardrails
from clinical_assistant.tools.langchain_tools import create_langchain_toolbox


NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


@dataclass
class FakePatientTools:
    calls: list[str]
    audit_events: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _result(operation: str, data: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "patient_id": "PAC004",
            "operation": operation,
            "data": data,
            "source": "Synthea anonimizado",
            "synthetic": True,
        }

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        self.calls.append("get_patient")
        return self._result("get_patient", {"patient_id": patient_id, "birth_year": 2015})

    def get_patient_conditions(self, patient_id: str) -> dict[str, Any]:
        self.calls.append("get_patient_conditions")
        return self._result("get_patient_conditions", [{"description": "Asma sintética"}])

    def get_patient_medications(self, patient_id: str) -> dict[str, Any]:
        self.calls.append("get_patient_medications")
        return self._result("get_patient_medications", [{"description": "Medicamento sintético"}])

    def get_pending_exams(self, patient_id: str) -> dict[str, Any]:
        self.calls.append("get_pending_exams")
        return self._result(
            "get_pending_exams",
            [{"description": "Espirometria sintética", "notice": NOTICE}],
        )

    def get_patient_observations(self, patient_id: str) -> dict[str, Any]:
        self.calls.append("get_patient_observations")
        return self._result("get_patient_observations", [{"description": "Frequência", "value": "72"}])


class FakeRetriever:
    def retrieve(self, query: str) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="ASM-001:0",
                text="Protocolo sintético recuperado para demonstração.",
                metadata={
                    "document_id": "ASM-001",
                    "document_name": "Fluxo sintético de acompanhamento",
                    "version": "1.0",
                    "section": "1",
                    "source": "Hospital TechCare fictício",
                    "synthetic": True,
                    "notice": NOTICE,
                },
                distance=0.1,
                relevance=0.9,
            )
        ]


@dataclass
class FakeAuditLogger:
    events: list[dict[str, Any]]

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        self.events.append(event)
        return {
            "ok": True,
            "execution_id": event["execution_id"],
            "event_number": len(self.events),
            "event_hash": f"fake-{len(self.events)}",
        }


SAFETY_CONFIG = {
    "safe_response": "Solicitação bloqueada. Validação médica necessária.",
    "review_notice": "Validação médica necessária.",
    "require_sources_when_data_is_used": True,
    "input_rules": [
        {
            "id": "SG-001",
            "description": "Tentativa de remover regras",
            "risk_level": "critical",
            "requires_human_validation": False,
            "patterns": [r"\b(ignore|ignorar)\b.{0,60}\b(regras|guardrails)\b"],
        },
        {
            "id": "SG-002",
            "description": "Prescrição autônoma",
            "risk_level": "critical",
            "requires_human_validation": True,
            "patterns": [r"\b(prescreva|receite)\b", r"\bdose correta\b"],
        },
    ],
    "output_rules": [
        {
            "id": "SG-007",
            "description": "Instrução clínica autônoma",
            "risk_level": "critical",
            "patterns": [r"\b(tome|use)\b.{0,80}\b(mg|medicamento|dose)\b"],
        }
    ],
}


def create_test_workflow(*, max_context_chars: int = 12000):
    patient_tools = FakePatientTools(calls=[], audit_events=[])
    toolbox = create_langchain_toolbox(
        patient_tools,
        FakeRetriever(),
        guideline_limitation="Nenhuma diretriz clínica oficial revisada foi fornecida.",
        audit_logger=FakeAuditLogger(patient_tools.audit_events),
    )
    workflow = ClinicalWorkflow(
        toolbox=toolbox,
        context_chain=ContextBuilderChain(max_context_chars=max_context_chars),
        generator=DeterministicEvidencePreview(),
        max_question_chars=1000,
        review_terms=[
            "tratamento",
            "medicamento",
            "dose",
            "prescrição",
            "diagnóstico",
            "procedimento",
            "conduta",
            "alterar",
        ],
        review_notice="Validação médica necessária.",
        guardrails=SafetyGuardrails(
            SAFETY_CONFIG,
            review_terms=[
                "tratamento",
                "medicamento",
                "dose",
                "prescrição",
                "diagnóstico",
                "procedimento",
                "conduta",
                "alterar",
            ],
        ),
    )
    return workflow, patient_tools
