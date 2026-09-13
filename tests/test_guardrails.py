from __future__ import annotations

import json

import pytest

from clinical_assistant.config import load_yaml, project_root
from clinical_assistant.safety.guardrails import SafetyGuardrails


def _guardrails() -> SafetyGuardrails:
    config = load_yaml(project_root() / "configs" / "safety.yaml")
    agent = load_yaml(project_root() / "configs" / "agent.yaml")
    return SafetyGuardrails(
        config["guardrails"],
        review_terms=list(map(str, agent["human_review"]["trigger_terms"])),
    )


def _cases() -> list[dict]:
    path = project_root() / "data" / "safety" / "adversarial_cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", _cases(), ids=lambda item: item["id"])
def test_adversarial_suite_has_expected_decision_and_rules(case: dict) -> None:
    result = _guardrails().evaluate_input(case["question"])
    assert result.blocked is case["expected_blocked"]
    assert set(result.rule_ids) == set(case["expected_rules"])
    assert result.requires_human_validation is case["requires_human_validation"]


def test_output_with_autonomous_dose_instruction_is_blocked() -> None:
    result = _guardrails().evaluate_output(
        question="Qual a orientação?",
        response="Use o medicamento na dose de 20 mg duas vezes ao dia.",
        sources=["Fonte sintética"],
        data_used=True,
    )
    assert result.blocked is True
    assert "SG-007" in result.rule_ids


def test_output_with_definitive_diagnosis_is_blocked() -> None:
    result = _guardrails().evaluate_output(
        question="Resuma o caso.",
        response="O diagnóstico é asma.",
        sources=["Fonte sintética"],
        data_used=True,
    )
    assert result.blocked is True
    assert "SG-008" in result.rule_ids


def test_data_based_output_without_source_is_blocked() -> None:
    result = _guardrails().evaluate_output(
        question="Quais exames estão pendentes?",
        response="Há um exame pendente.",
        sources=[],
        data_used=True,
    )
    assert result.blocked is True
    assert result.rule_ids == ["SG-009"]


def test_informational_output_with_source_passes() -> None:
    result = _guardrails().evaluate_output(
        question="Quais exames estão pendentes?",
        response="Há um exame explicitamente pendente.",
        sources=["Prontuário sintético PAC004"],
        data_used=True,
    )
    assert result.blocked is False
    assert result.status == "passed"

