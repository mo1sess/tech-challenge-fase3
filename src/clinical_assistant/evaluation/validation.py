"""Local preflight and evidence validation for the stage 10 comparison."""

from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.evaluation.dataset import load_evaluation_cases, validate_evaluation_dataset


def _normalized_lf_sha256(path: Path) -> str:
    """Hash text with LF line endings so Windows and Colab evidence can be compared."""

    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append(f"invalid JSON {label}: {path}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    return value


def validate_stage10_preflight(root: Path, *, write_report: bool = True) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "model_evaluation.yaml")
    errors: list[str] = []
    paths = {
        key: resolve_project_path(value, root=root)
        for key, value in config["prerequisites"].items()
    }
    baseline = _read_object(paths["baseline_summary"], errors, "baseline summary")
    training = _read_object(paths["training_manifest"], errors, "training manifest")
    rag = _read_object(paths["rag_manifest"], errors, "RAG manifest")
    baseline_response_hash: str | None = None
    baseline_response_count = 0
    baseline_records: list[dict[str, Any]] = []
    if not paths["baseline_responses"].is_file():
        errors.append(f"missing baseline responses: {paths['baseline_responses']}")
    else:
        baseline_response_hash = _normalized_lf_sha256(paths["baseline_responses"])
        try:
            baseline_records = [
                json.loads(line)
                for line in paths["baseline_responses"].read_text(encoding="utf-8").splitlines()
                if line
            ]
        except json.JSONDecodeError:
            baseline_records = []
            errors.append("baseline responses contain invalid JSONL")
        baseline_response_count = len(baseline_records)

    expected_cases = int(config["execution"]["expected_cases"])
    if baseline.get("stage") != 4 or baseline.get("complete") is not True:
        errors.append("official stage 4 baseline is incomplete")
    if baseline.get("cases") != expected_cases:
        errors.append("baseline case count does not match stage 10")
    if training.get("stage") != 5 or training.get("complete") is not True:
        errors.append("official stage 5 training is incomplete")
    if rag.get("stage") != 6 or rag.get("complete") is not True:
        errors.append("stage 6 RAG prerequisite is incomplete")

    variants = list(map(str, config.get("variants", [])))
    if variants != ["base", "fine_tuned", "fine_tuned_rag"]:
        errors.append("required comparison variants are not exact")
    if config["model"].get("allow_substitution") is not False:
        errors.append("model substitution must remain disabled")
    configured_model = (config["model"]["id"], config["model"]["revision"])
    for label, manifest in (("baseline", baseline), ("training", training)):
        model = manifest.get("model", {})
        if (model.get("id"), model.get("revision")) != configured_model:
            errors.append(f"{label} model/revision differs from stage 10 configuration")

    baseline_config = load_yaml(root / "configs" / "baseline.yaml")
    if config["generation"] != baseline_config["generation"]:
        errors.append("generation settings must match the official baseline")
    dataset = validate_evaluation_dataset(root)
    if not dataset["ok"]:
        errors.extend(f"evaluation dataset: {error}" for error in dataset["errors"])
    normalized_hashes = {
        filename: _normalized_lf_sha256(
            resolve_project_path(config["execution"]["evaluation_directory"], root=root)
            / filename
        )
        for filename in baseline_config["evaluation"]["required_files"]
    }
    if baseline.get("evaluation_dataset_sha256") != normalized_hashes:
        errors.append("held-out evaluation hashes differ from the official baseline")
    try:
        expected_ids = {
            str(case["id"])
            for case in load_evaluation_cases(
                resolve_project_path(
                    config["execution"]["evaluation_directory"], root=root
                ),
                list(map(str, baseline_config["evaluation"]["required_files"])),
            )
        }
    except (FileNotFoundError, ValueError) as exc:
        expected_ids = set()
        errors.append(f"cannot load held-out case identifiers: {exc}")
    response_ids = {str(record.get("id")) for record in baseline_records}
    if baseline_response_count != expected_cases or response_ids != expected_ids:
        errors.append("baseline responses do not contain the exact held-out cases")
    if baseline.get("responses_sha256") != baseline_response_hash:
        errors.append("baseline response hash differs from the official summary")
    adapter_hash = training.get("artifact_sha256", {}).get(
        "adapter/adapter_model.safetensors"
    )
    if not isinstance(adapter_hash, str) or len(adapter_hash) != 64:
        errors.append("official adapter SHA-256 is missing")

    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "stage": 10,
        "status": "ready_for_remote_gpu_evaluation" if not errors else "blocked",
        "variants": variants,
        "cases_per_variant": expected_cases,
        "total_expected_generations": expected_cases * 2,
        "model": {"id": configured_model[0], "revision": configured_model[1]},
        "adapter_sha256": adapter_hash,
        "evaluation_dataset_sha256_lf": normalized_hashes,
        "baseline_responses_sha256_lf": baseline_response_hash,
        "prerequisites": {
            "baseline_complete": baseline.get("complete"),
            "training_complete": training.get("complete"),
            "rag_complete": rag.get("complete"),
            "evaluation_dataset_ok": dataset.get("ok"),
        },
        "execution": {
            "preferred": config["execution"]["preferred"],
            "minimum_vram_gb": config["execution"]["minimum_vram_gb"],
            "local_gpu_allowed": config["execution"]["local_gpu_allowed"],
            "gpu_used_by_preflight": False,
        },
        "errors": errors,
    }
    if write_report:
        write_json(
            resolve_project_path(config["outputs"]["preflight_report"], root=root),
            report,
        )
    return report


def validate_evaluation_run(root: Path, run_directory: Path) -> dict[str, Any]:
    """Validate imported Colab evidence without recalculating model outputs."""

    config = load_yaml(root / "configs" / "model_evaluation.yaml")
    errors: list[str] = []
    run_directory = (
        run_directory if run_directory.is_absolute() else root / run_directory
    ).resolve()
    output_root = resolve_project_path(config["execution"]["output_directory"], root=root)
    if run_directory != output_root and output_root not in run_directory.parents:
        raise ValueError("evaluation run must be inside outputs/evaluation")
    summary = _read_object(run_directory / "summary.json", errors, "evaluation summary")
    if not (run_directory / "comparison.md").is_file():
        errors.append("missing comparison.md")
    expected = int(config["execution"]["expected_cases"])
    expected_ids: set[str] = set()
    evaluation_config = load_yaml(root / "configs" / "baseline.yaml")["evaluation"]
    for filename in evaluation_config["required_files"]:
        path = root / "data" / "evaluation" / str(filename)
        expected_ids.update(
            str(json.loads(line)["id"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    response_counts: dict[str, int] = {}
    for variant in ("fine_tuned", "fine_tuned_rag"):
        path = run_directory / f"{variant}_responses.jsonl"
        if not path.is_file():
            errors.append(f"missing {variant} responses")
            continue
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except json.JSONDecodeError:
            errors.append(f"invalid JSONL for {variant}")
            continue
        response_counts[variant] = len(records)
        identifiers = {str(item.get("id")) for item in records}
        if len(records) != expected or identifiers != expected_ids:
            errors.append(f"{variant} does not contain the exact {expected} held-out cases")
        recorded_hash = summary.get("variants", {}).get(variant, {}).get("responses_sha256")
        if recorded_hash != sha256_file(path):
            errors.append(f"{variant} response hash mismatch")
    if summary.get("stage") != 10 or summary.get("complete") is not True:
        errors.append("stage 10 summary is not an official complete run")
    if summary.get("run_type") != "official_full_model_comparison":
        errors.append("stage 10 run_type is not official_full_model_comparison")
    if list(summary.get("variants", {})) != ["base", "fine_tuned", "fine_tuned_rag"]:
        errors.append("stage 10 summary variants are incomplete or out of order")
    return {
        "ok": not errors,
        "stage": 10,
        "run_directory": str(run_directory),
        "complete": summary.get("complete"),
        "response_counts": response_counts,
        "summary_sha256": sha256_file(run_directory / "summary.json")
        if (run_directory / "summary.json").is_file()
        else None,
        "errors": errors,
    }
