from __future__ import annotations

import pytest

from clinical_assistant.chains.context_chain import ContextBuilderChain
from clinical_assistant.tools.langchain_tools import create_langchain_toolbox
from tests.agent_helpers import FakePatientTools, FakeRetriever


@pytest.mark.unit
def test_toolbox_exposes_only_named_bounded_tools() -> None:
    toolbox = create_langchain_toolbox(
        FakePatientTools(calls=[]),
        FakeRetriever(),
        guideline_limitation="Fonte oficial não fornecida.",
    )
    names = set(toolbox.by_name())
    assert names == {
        "get_patient",
        "get_patient_conditions",
        "get_patient_medications",
        "get_pending_exams",
        "get_patient_observations",
        "search_internal_protocol",
        "search_clinical_guideline",
    }
    assert "execute_sql" not in names
    assert "save_audit_log" not in names


@pytest.mark.unit
def test_internal_protocol_tool_returns_citation_and_synthetic_notice() -> None:
    toolbox = create_langchain_toolbox(
        FakePatientTools(calls=[]),
        FakeRetriever(),
        guideline_limitation="Fonte oficial não fornecida.",
    )
    result = toolbox.by_name()["search_internal_protocol"].invoke(
        {"query": "Como acompanhar asma?"}
    )
    assert result["documents"][0]["metadata"]["document_id"] == "ASM-001"
    assert "fonte:" in result["citations"][0]
    assert result["synthetic"] is True


@pytest.mark.unit
def test_guideline_tool_is_transparent_when_official_source_is_absent() -> None:
    toolbox = create_langchain_toolbox(
        FakePatientTools(calls=[]),
        FakeRetriever(),
        guideline_limitation="Fonte oficial não fornecida.",
    )
    result = toolbox.by_name()["search_clinical_guideline"].invoke(
        {"query": "Qual é o PCDT?"}
    )
    assert result["ok"] is False
    assert result["documents"] == []
    assert result["limitation"] == "Fonte oficial não fornecida."


@pytest.mark.unit
def test_context_chain_builds_evidence_bound_prompt_and_limits_context() -> None:
    chain = ContextBuilderChain(max_context_chars=1000)
    result = chain.invoke(
        {
            "patient_id": "PAC004",
            "question": "Quais exames estão pendentes?",
            "patient_data": {"patient": {"patient_id": "PAC004", "value": "x" * 2000}},
            "pending_exams": [],
            "retrieved_documents": [],
            "citations": [],
        }
    )
    assert len(result["context"]) < 1100
    assert "CONTEXTO TRUNCADO" in result["context"]
    assert "Não invente dados" in result["prompt"]
    assert "Nenhuma fonte clínica oficial" in result["prompt"]
