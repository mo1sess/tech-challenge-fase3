from __future__ import annotations

import pytest

from clinical_assistant.graph.response_consistency import enforce_response_consistency


def _state(question: str) -> dict:
    return {
        "patient_id": "PAC006",
        "question": question,
        "patient_data": {
            "medications": [
                {
                    "code": "861467",
                    "description": "Meperidine Hydrochloride 50 MG Oral Tablet",
                    "reason_description": None,
                }
            ],
            "conditions": [
                {
                    "code_system": "SNOMED-CT",
                    "code": "195967001",
                    "description": "Asthma (disorder)",
                }
            ],
            "observations": [
                {
                    "observed_at": "2026-09-01T10:00:00Z",
                    "code": "8867-4",
                    "description": "Heart rate",
                    "value": "72",
                    "units": "beats/min",
                }
            ],
        },
        "pending_exams": [
            {
                "exam_request_id": "PEX-001",
                "exam_code": "DEMO-EXAM-A",
                "description": "Solicitação sintética de exame A",
                "requested_at": "2026-09-01",
                "due_at": "2026-09-10",
                "status": "pending",
                "source": "Hospital TechCare",
            }
        ],
        "retrieved_documents": [],
    }


@pytest.mark.unit
def test_medication_lookup_replaces_model_hallucination_with_literal_record() -> None:
    response, report = enforce_response_consistency(
        "O paciente utiliza piretanida 50 mg.",
        _state("Quais medicamentos aparecem no prontuário?"),
    )
    assert "Meperidine Hydrochloride 50 MG Oral Tablet" in response
    assert "861467" in response
    assert "motivo registrado: não informado" in response
    assert "piretanida" not in response.casefold()
    assert report["applied"] is True
    assert report["reason"] == "factual_medications"


@pytest.mark.unit
def test_unfilled_placeholder_is_never_released() -> None:
    response, report = enforce_response_consistency(
        "Modelo: exame [CÓDIGO]; data [DATA].",
        _state("Faça um resumo do prontuário."),
    )
    assert "[CÓDIGO]" not in response
    assert "[DATA]" not in response
    assert report["reason"] == "unfilled_model_placeholder"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "literal", "reason"),
    [
        ("Quais condições estão registradas?", "Asthma (disorder)", "factual_conditions"),
        ("Quais observações aparecem no prontuário?", "72 beats/min", "factual_observations"),
        ("Quais exames estão pendentes?", "DEMO-EXAM-A", "factual_pending_exams"),
    ],
)
def test_other_factual_queries_use_literal_structured_evidence(
    question: str, literal: str, reason: str
) -> None:
    response, report = enforce_response_consistency("Texto livre incorreto.", _state(question))
    assert literal in response
    assert report["reason"] == reason
    assert report["applied"] is True


@pytest.mark.unit
def test_treatment_question_is_not_misclassified_as_record_lookup() -> None:
    candidate = "Não altere medicamentos sem validação médica."
    response, report = enforce_response_consistency(
        candidate,
        _state("Devo alterar o medicamento?"),
    )
    assert response == candidate
    assert report["applied"] is False


@pytest.mark.unit
def test_non_factual_model_response_is_retained_when_it_has_no_placeholder() -> None:
    candidate = "O protocolo sintético recuperado exige validação médica."
    response, report = enforce_response_consistency(
        candidate,
        _state("Explique o protocolo recuperado."),
    )
    assert response == candidate
    assert report["applied"] is False
