from __future__ import annotations

import pytest
from langgraph.types import Command

from tests.agent_helpers import create_test_workflow


def _initial(question: str) -> dict:
    return {
        "patient_id": "PAC004",
        "question": question,
        "execution_id": "test-execution",
        "trace": [],
    }


@pytest.mark.integration
def test_informational_flow_finishes_without_human_interrupt() -> None:
    workflow, patient_tools = create_test_workflow()
    graph = workflow.compile()
    result = graph.invoke(
        _initial("Quais exames estão pendentes?"),
        config={"configurable": {"thread_id": "informational"}},
    )
    assert "__interrupt__" not in result
    assert result["requires_human_validation"] is False
    assert result["trace"] == [
        "validate_input",
        "load_patient",
        "check_pending_exams",
        "retrieve_protocols",
        "build_context",
        "generate_response",
        "safety_check",
        "finalize_response",
    ]
    assert patient_tools.calls == ["get_patient", "get_pending_exams"]
    assert "Fontes recuperadas" in result["final_response"]


@pytest.mark.integration
def test_clinical_flow_interrupts_and_can_be_rejected() -> None:
    workflow, patient_tools = create_test_workflow()
    graph = workflow.compile()
    config = {"configurable": {"thread_id": "clinical-reject"}}
    paused = graph.invoke(_initial("Devo alterar o medicamento?"), config=config)
    assert "__interrupt__" in paused
    assert paused["requires_human_validation"] is True
    assert "medicamento" in paused["safety_result"]["triggers"]
    assert "get_patient_medications" in patient_tools.calls

    final = graph.invoke(
        Command(resume={"approved": False, "reviewer": "test", "notes": "Rejeitado"}),
        config=config,
    )
    assert final["human_validation"]["approved"] is False
    assert "Resposta retida" in final["final_response"]
    assert final["trace"][-2:] == ["human_review", "finalize_response"]


@pytest.mark.integration
def test_clinical_flow_can_resume_after_approval() -> None:
    workflow, _ = create_test_workflow()
    graph = workflow.compile()
    config = {"configurable": {"thread_id": "clinical-approve"}}
    paused = graph.invoke(_initial("Qual procedimento devo iniciar?"), config=config)
    assert "__interrupt__" in paused

    final = graph.invoke(
        Command(resume={"approved": True, "reviewer": "test", "notes": "Demo"}),
        config=config,
    )
    assert final["human_validation"]["approved"] is True
    assert "Validação médica necessária" in final["final_response"]
    assert "não comprova revisão médica profissional" in final["final_response"]


@pytest.mark.unit
def test_graph_rejects_invalid_input_before_accessing_tools() -> None:
    workflow, patient_tools = create_test_workflow()
    graph = workflow.compile()
    with pytest.raises(ValueError, match="PACnnn"):
        graph.invoke(
            _initial("Pergunta") | {"patient_id": "PAC004' OR 1=1 --"},
            config={"configurable": {"thread_id": "invalid"}},
        )
    assert patient_tools.calls == []


@pytest.mark.unit
def test_graph_has_a_real_conditional_edge() -> None:
    workflow, _ = create_test_workflow()
    graph_view = workflow.compile().get_graph()
    safety_edges = [edge for edge in graph_view.edges if edge.source == "safety_check"]
    assert {edge.target for edge in safety_edges} == {"human_review", "finalize_response"}
    assert all(edge.conditional for edge in safety_edges)
