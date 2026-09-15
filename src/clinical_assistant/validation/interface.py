"""Validate the local Streamlit MVP and its prerequisite artifacts."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.database.patient_repository import PatientRepository


EXPECTED_STREAMLIT_VERSION = "1.46.1"
REQUIRED_SECTIONS = {
    "patient_selector",
    "question_input",
    "answer",
    "sources",
    "pending_exams",
    "safety_status",
    "human_validation",
    "audit",
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {path} ({exc})")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object: {path}")
        return {}
    return value


def validate_streamlit_stage(root: Path, *, write_report: bool = True) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "interface.yaml")
    errors: list[str] = []

    try:
        streamlit_version = importlib.metadata.version("streamlit")
    except importlib.metadata.PackageNotFoundError:
        streamlit_version = None
    if streamlit_version != EXPECTED_STREAMLIT_VERSION:
        errors.append(
            f"streamlit: expected {EXPECTED_STREAMLIT_VERSION}, found {streamlit_version}"
        )

    if config.get("stage") != 11:
        errors.append("interface configuration must declare stage 11")
    execution = config.get("execution", {})
    if execution.get("mode") != "local_cpu":
        errors.append("the Streamlit MVP must remain local_cpu")
    if execution.get("generator") != "deterministic_evidence_preview":
        errors.append("local UI must identify the deterministic preview generator")
    if execution.get("official_model_local_loading") is not False:
        errors.append("the official Qwen3-8B cannot be loaded on the local GTX 1650")

    sections = set(map(str, config.get("interface", {}).get("required_sections", [])))
    missing_sections = sorted(REQUIRED_SECTIONS - sections)
    if missing_sections:
        errors.append(f"missing required UI sections: {missing_sections}")

    prerequisites = config.get("prerequisites", {})
    artifact_specs = (
        ("rag_manifest", 6, "complete"),
        ("database_manifest", 7, "complete"),
        ("agent_report", 8, "ok"),
        ("safety_report", 9, "ok"),
        ("evaluation_summary", 10, "complete"),
    )
    artifact_status: dict[str, Any] = {}
    for key, expected_stage, completion_key in artifact_specs:
        path = resolve_project_path(prerequisites[key], root=root)
        value = _read_json(path, errors, key)
        valid = value.get("stage") == expected_stage and value.get(completion_key) is True
        artifact_status[key] = {
            "path": _relative(path, root),
            "stage": value.get("stage"),
            "ready": valid,
        }
        if value and not valid:
            errors.append(f"{key} is not a completed stage {expected_stage} artifact")

    entrypoint = resolve_project_path(prerequisites["app_entrypoint"], root=root)
    if not entrypoint.is_file():
        errors.append(f"missing Streamlit entrypoint: {entrypoint}")

    database_path = resolve_project_path(prerequisites["database_path"], root=root)
    patient_count = 0
    if not database_path.is_file():
        errors.append(f"missing local patient database: {database_path}")
    else:
        database_config = load_yaml(root / "configs" / "database.yaml")["database"]
        repository = PatientRepository(
            database_path,
            maximum_result_limit=int(database_config["maximum_result_limit"]),
        )
        patient_count = len(repository.list_patient_ids())
        if patient_count < 1:
            errors.append("patient selector has no pseudonymized patients")

    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "stage": 11,
        "streamlit": streamlit_version,
        "entrypoint": _relative(entrypoint, root),
        "patient_count": patient_count,
        "required_sections": sorted(sections),
        "prerequisites": artifact_status,
        "execution": {
            "mode": execution.get("mode"),
            "generator": execution.get("generator"),
            "official_model": execution.get("official_model"),
            "official_model_loaded_locally": False,
            "gpu_used": False,
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
