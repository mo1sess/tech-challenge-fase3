from __future__ import annotations

from dataclasses import dataclass, field
import json

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.interface.application import ClinicalApplication
from clinical_assistant.validation.interface import validate_streamlit_stage
from tests.agent_helpers import create_test_workflow


class HallucinatingGenerator:
    mode = "qwen3_8b_qlora_remote"

    def generate(self, prompt: str, state: dict) -> str:
        return "Piretanida 50 mg e exame [CÓDIGO] em [DATA]."


@dataclass
class FakePatientRepository:
    patient_ids: list[str] = field(default_factory=lambda: ["PAC004", "PAC001"])

    def list_patient_ids(self) -> list[str]:
        return list(self.patient_ids)


@dataclass
class FakeAuditReader:
    events: list[dict] = field(default_factory=list)

    def verify(self) -> dict:
        return {
            "ok": True,
            "events": len(self.events),
            "last_hash": "0" * 64,
            "path": "fake-audit.jsonl",
        }

    def read_events(self, *, limit: int) -> list[dict]:
        return self.events[-limit:]


def _application() -> ClinicalApplication:
    workflow, _ = create_test_workflow()
    return ClinicalApplication(
        graph=workflow.compile(),
        patient_repository=FakePatientRepository(),
        audit_logger=FakeAuditReader(),
    )


@pytest.mark.unit
def test_patient_selector_is_sorted_and_pseudonymized() -> None:
    application = _application()
    assert application.list_patient_ids() == ["PAC001", "PAC004"]


@pytest.mark.integration
def test_informational_query_produces_all_required_view_sections() -> None:
    result = _application().submit_question(
        "PAC004", "Quais exames estão pendentes?", thread_id="streamlit-info"
    )
    assert result["status"] == "completed"
    assert result["answer"]
    assert "Fontes recuperadas:" not in result["answer"]
    assert "Dados sintéticos para demonstração" not in result["answer"]
    assert result["sources"]
    assert len(result["pending_exams"]) == 1
    assert result["safety_label"] == "Consulta informativa"
    assert result["requires_human_validation"] is False
    assert result["audit_result"]["ok"] is True


@pytest.mark.integration
def test_condition_query_lists_retrieved_conditions() -> None:
    result = _application().submit_question(
        "PAC001", "Quais condições estão registradas?", thread_id="streamlit-conditions"
    )
    assert result["status"] == "completed"
    assert "Condições registradas" in result["answer"]
    assert "Asma sintética" in result["answer"]
    assert "baseada apenas nos dados recuperados" not in result["answer"]
    assert result["safety_label"] == "Consulta informativa"


@pytest.mark.integration
def test_factual_medication_answer_is_locked_to_sqlite_evidence() -> None:
    workflow, _ = create_test_workflow(generator=HallucinatingGenerator())
    application = ClinicalApplication(
        graph=workflow.compile(),
        patient_repository=FakePatientRepository(),
        audit_logger=FakeAuditReader(),
    )
    paused = application.submit_question(
        "PAC004", "Quais medicamentos aparecem no prontuário?", thread_id="locked-medications"
    )
    assert paused["status"] == "awaiting_human_review"
    assert "Medicamento sintético" in paused["draft"]
    assert "Piretanida" not in paused["draft"]
    assert "[CÓDIGO]" not in paused["draft"]
    assert paused["generator_mode"].endswith("+evidence_lock")
    assert paused["response_consistency"]["reason"] == "factual_medications"
    assert paused["structured_evidence"] == [{"description": "Medicamento sintético"}]


@pytest.mark.integration
def test_human_review_can_be_resumed_from_the_ui_controller() -> None:
    application = _application()
    paused = application.submit_question(
        "PAC004", "Devo alterar o medicamento?", thread_id="streamlit-review"
    )
    assert paused["status"] == "awaiting_human_review"
    assert paused["requires_human_validation"] is True
    assert paused["draft"]
    assert not paused["answer"]

    final = application.resume_review(
        paused["thread_id"], approved=False, notes="Rejeitado na demonstração"
    )
    assert final["status"] == "completed"
    assert final["human_validation"]["approved"] is False
    assert "Resposta retida" in final["answer"]


@pytest.mark.integration
def test_stage11_validator_confirms_local_mvp() -> None:
    result = validate_streamlit_stage(project_root(), write_report=False)
    assert result["ok"] is True
    assert result["stage"] == 11
    assert result["patient_count"] == 108
    assert result["streamlit"] == "1.46.1"
    assert result["execution"]["official_model_loaded_locally"] is False
    assert str(project_root()) not in json.dumps(result)


@pytest.mark.unit
def test_streamlit_entrypoint_contains_required_visible_labels() -> None:
    source = (project_root() / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    compile(source, "streamlit_app.py", "exec")
    for label in (
        "Paciente",
        "Pergunta",
        "Consultar",
        "Resposta",
        "Fontes",
        "Exames pendentes",
        "Status de segurança",
        "Validação médica necessária",
        "Auditoria e log",
    ):
        assert label in source
