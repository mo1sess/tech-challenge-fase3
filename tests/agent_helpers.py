from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clinical_assistant.chains.context_chain import ContextBuilderChain
from clinical_assistant.graph.workflow import ClinicalWorkflow, DeterministicEvidencePreview
from clinical_assistant.rag.models import RetrievalHit
from clinical_assistant.tools.langchain_tools import create_langchain_toolbox


NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


@dataclass
class FakePatientTools:
    calls: list[str]

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


def create_test_workflow(*, max_context_chars: int = 12000):
    patient_tools = FakePatientTools(calls=[])
    toolbox = create_langchain_toolbox(
        patient_tools,
        FakeRetriever(),
        guideline_limitation="Nenhuma diretriz clínica oficial revisada foi fornecida.",
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
    )
    return workflow, patient_tools
