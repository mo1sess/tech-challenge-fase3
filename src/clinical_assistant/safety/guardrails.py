"""Deterministic input/output guardrails with explicit rule evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def normalize_for_safety(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(char for char in decomposed if not unicodedata.combining(char)).split()
    )


@dataclass(frozen=True)
class SafetyAssessment:
    stage: str
    status: str
    blocked: bool
    requires_human_validation: bool
    risk_level: str
    rule_ids: list[str]
    rule_descriptions: list[str]
    sources_present: bool
    message: str
    safe_response: str | None
    checked_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyGuardrails:
    def __init__(self, config: dict[str, Any], *, review_terms: list[str]) -> None:
        self.config = config
        self.review_terms = [normalize_for_safety(term) for term in review_terms]
        self.safe_response = str(config["safe_response"])
        self.require_sources = bool(config["require_sources_when_data_is_used"])
        self.input_rules = self._compile_rules(config["input_rules"])
        self.output_rules = self._compile_rules(config["output_rules"])

    @staticmethod
    def _compile_rules(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compiled: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            identifier = str(value["id"])
            if identifier in seen:
                raise ValueError(f"duplicate safety rule: {identifier}")
            seen.add(identifier)
            compiled.append(
                {
                    **value,
                    "patterns": [
                        re.compile(normalize_for_safety(str(pattern)), re.IGNORECASE)
                        for pattern in value["patterns"]
                    ],
                }
            )
        return compiled

    @staticmethod
    def _matched_rules(text: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = normalize_for_safety(text)
        return [rule for rule in rules if any(pattern.search(normalized) for pattern in rule["patterns"])]

    @staticmethod
    def _risk(matches: list[dict[str, Any]]) -> str:
        return max(
            (str(rule["risk_level"]) for rule in matches),
            key=lambda value: RISK_ORDER[value],
            default="low",
        )

    def evaluate_input(self, question: str) -> SafetyAssessment:
        matches = self._matched_rules(question, self.input_rules)
        blocked = bool(matches)
        requires_review = any(
            bool(rule.get("requires_human_validation")) for rule in matches
        )
        return SafetyAssessment(
            stage="input",
            status="blocked" if blocked else "passed",
            blocked=blocked,
            requires_human_validation=requires_review,
            risk_level=self._risk(matches),
            rule_ids=[str(rule["id"]) for rule in matches],
            rule_descriptions=[str(rule["description"]) for rule in matches],
            sources_present=False,
            message=(
                "Solicitação bloqueada por regra determinística de segurança."
                if blocked
                else "Entrada permitida para processamento controlado."
            ),
            safe_response=self.safe_response if blocked else None,
            checked_at_utc=datetime.now(UTC).isoformat(),
        )

    def evaluate_output(
        self,
        *,
        question: str,
        response: str,
        sources: list[str],
        data_used: bool,
    ) -> SafetyAssessment:
        matches = self._matched_rules(response, self.output_rules)
        missing_sources = self.require_sources and data_used and not sources
        blocked = bool(matches) or missing_sources
        review_requested = any(
            term in normalize_for_safety(question) for term in self.review_terms
        )
        rule_ids = [str(rule["id"]) for rule in matches]
        descriptions = [str(rule["description"]) for rule in matches]
        if missing_sources:
            rule_ids.append("SG-009")
            descriptions.append("Resposta baseada em dados sem fonte apresentada")
        requires_review = review_requested and not blocked
        status = "blocked" if blocked else (
            "human_review_required" if requires_review else "passed"
        )
        return SafetyAssessment(
            stage="output",
            status=status,
            blocked=blocked,
            requires_human_validation=requires_review,
            risk_level="high" if missing_sources and not matches else self._risk(matches),
            rule_ids=rule_ids,
            rule_descriptions=descriptions,
            sources_present=bool(sources),
            message=(
                "Resposta bloqueada pela validação de saída."
                if blocked
                else "Resposta exige revisão humana."
                if requires_review
                else "Resposta aprovada pelos controles determinísticos."
            ),
            safe_response=self.safe_response if blocked else None,
            checked_at_utc=datetime.now(UTC).isoformat(),
        )
