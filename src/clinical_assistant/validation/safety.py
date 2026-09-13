"""Validate the deterministic safety and audit controls introduced in stage 9."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.safety.guardrails import SafetyGuardrails


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing prerequisite report: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, f"invalid JSON prerequisite report: {path}"
    if not isinstance(value, dict):
        return {}, f"prerequisite report is not an object: {path}"
    return value, None


def _read_cases(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return cases, [f"missing adversarial suite: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid adversarial JSON at line {line_number}")
            continue
        if not isinstance(value, dict):
            errors.append(f"adversarial line {line_number} is not an object")
            continue
        cases.append(value)
    return cases, errors


def validate_safety_stage(root: Path, *, write_report: bool = True) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "safety.yaml")
    agent_config = load_yaml(root / "configs" / "agent.yaml")
    errors: list[str] = []

    previous, prerequisite_error = _read_json(
        root / "outputs" / "agent" / "stage8_validation.json"
    )
    if prerequisite_error:
        errors.append(prerequisite_error)
    elif previous.get("stage") != 8 or previous.get("ok") is not True:
        errors.append("prerequisite stage 8 is not valid")

    try:
        guardrails = SafetyGuardrails(
            config["guardrails"],
            review_terms=list(map(str, agent_config["human_review"]["trigger_terms"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        guardrails = None
        errors.append(f"invalid guardrail configuration: {exc}")

    suite_path = resolve_project_path(config["adversarial_suite"]["path"], root=root)
    cases, case_errors = _read_cases(suite_path)
    errors.extend(case_errors)
    identifiers = [str(case.get("id", "")) for case in cases]
    if len(identifiers) != len(set(identifiers)):
        errors.append("adversarial case identifiers are not unique")
    expected_count = int(config["adversarial_suite"]["expected_cases"])
    if len(cases) != expected_count:
        errors.append(f"expected {expected_count} adversarial cases, found {len(cases)}")

    results: list[dict[str, Any]] = []
    if guardrails is not None:
        for case in cases:
            missing = {
                "id",
                "question",
                "expected_blocked",
                "expected_rules",
                "requires_human_validation",
            } - set(case)
            if missing:
                errors.append(f"case {case.get('id', '?')} missing fields: {sorted(missing)}")
                continue
            assessment = guardrails.evaluate_input(str(case["question"]))
            matched = (
                assessment.blocked is case["expected_blocked"]
                and set(assessment.rule_ids) == set(map(str, case["expected_rules"]))
                and assessment.requires_human_validation
                is case["requires_human_validation"]
            )
            if not matched:
                errors.append(f"unexpected guardrail result for {case['id']}")
            results.append(
                {
                    "id": case["id"],
                    "matched_expectation": matched,
                    "blocked": assessment.blocked,
                    "rule_ids": assessment.rule_ids,
                    "requires_human_validation": assessment.requires_human_validation,
                }
            )

    audit = config.get("audit", {})
    if audit.get("format") != "jsonl_hash_chain_v1":
        errors.append("audit format must be jsonl_hash_chain_v1")
    if audit.get("append_only") is not True:
        errors.append("audit must remain append-only")
    try:
        audit_path = resolve_project_path(str(audit["path"]), root=root)
    except (KeyError, ValueError) as exc:
        audit_path = root
        errors.append(f"invalid audit path: {exc}")

    graph_nodes = list(map(str, config["expected"]["graph_nodes"]))
    conditional_edges = list(map(str, config["expected"]["conditional_edges"]))
    if len(graph_nodes) != len(set(graph_nodes)):
        errors.append("graph node names are not unique")
    if not set(conditional_edges).issubset(graph_nodes):
        errors.append("conditional edge references an unknown graph node")
    required_nodes = {"input_safety", "safety_check", "human_review", "audit_log"}
    if not required_nodes.issubset(graph_nodes):
        errors.append("stage 9 safety/audit graph nodes are incomplete")

    covered_rules = sorted(
        {rule for result in results for rule in result.get("rule_ids", [])}
    )
    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "stage": 9,
        "prerequisite": {"stage": previous.get("stage"), "ok": previous.get("ok")},
        "guardrails": {
            "cases": len(results),
            "blocked": sum(bool(item["blocked"]) for item in results),
            "allowed": sum(not bool(item["blocked"]) for item in results),
            "matched_expectations": sum(bool(item["matched_expectation"]) for item in results),
            "covered_input_rules": covered_rules,
            "results": results,
        },
        "audit": {
            "format": audit.get("format"),
            "append_only": audit.get("append_only"),
            "path": str(audit_path.relative_to(root.resolve())),
            "hash_chain": "sha256",
            "runtime_log_versioned": False,
        },
        "graph": {
            "nodes": graph_nodes,
            "node_count": len(graph_nodes),
            "conditional_edges": conditional_edges,
        },
        "execution": {
            "mode": "local_cpu",
            "gpu_used": False,
            "official_qwen_executed": False,
        },
        "limitations": [
            "Regras determinísticas não comprovam correção clínica.",
            "O JSONL local com hash detecta adulteração, mas não é um sistema regulatório.",
            "Decisões clínicas continuam exigindo validação profissional humana.",
        ],
        "errors": errors,
    }
    if write_report:
        report_path = resolve_project_path(config["outputs"]["validation_report"], root=root)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report

