"""Validate stage 8 configuration, dependencies, and prerequisite artifacts."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path


EXPECTED_VERSIONS = {
    "langchain": "0.3.26",
    "langchain-community": "0.3.26",
    "langgraph": "0.4.8",
}


def _read_manifest(path: Path, stage: int) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing prerequisite manifest: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, f"invalid prerequisite manifest: {path}"
    if value.get("stage") != stage or value.get("complete") is not True:
        return value, f"prerequisite stage {stage} is not complete"
    return value, None


def validate_agent_stage(root: Path, *, write_report: bool = True) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "agent.yaml")
    errors: list[str] = []
    versions: dict[str, str | None] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[package] = actual
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")

    rag_manifest, rag_error = _read_manifest(root / "outputs" / "rag" / "index_manifest.json", 6)
    database_manifest, database_error = _read_manifest(
        root / "outputs" / "database" / "database_manifest.json", 7
    )
    errors.extend(error for error in (rag_error, database_error) if error)

    required_nodes = list(map(str, config["expected"]["graph_nodes"]))
    if len(required_nodes) != len(set(required_nodes)):
        errors.append("graph node names are not unique")
    if config["expected"]["conditional_edge"] not in required_nodes:
        errors.append("configured conditional edge is not a graph node")
    if config["execution"]["mode"] != "local_cpu":
        errors.append("stage 8 execution mode must remain local_cpu")
    safety = config["safety"]
    if safety["allow_autonomous_prescription"] or safety["allow_definitive_diagnosis"]:
        errors.append("unsafe autonomous clinical behavior enabled")
    if safety["allow_arbitrary_sql"]:
        errors.append("arbitrary SQL must remain disabled")

    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "stage": 8,
        "dependencies": versions,
        "prerequisites": {
            "rag_stage": rag_manifest.get("stage"),
            "rag_documents": rag_manifest.get("source", {}).get("logical_documents"),
            "database_stage": database_manifest.get("stage"),
            "database_patients": database_manifest.get("table_counts", {}).get("patients"),
        },
        "graph": {
            "nodes": required_nodes,
            "node_count": len(required_nodes),
            "conditional_edge": config["expected"]["conditional_edge"],
            "human_review_enabled": bool(config["human_review"]["enabled"]),
        },
        "execution": {
            "mode": config["execution"]["mode"],
            "generator": config["execution"]["generator"],
            "gpu_used": False,
            "official_qwen_executed": False,
        },
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
