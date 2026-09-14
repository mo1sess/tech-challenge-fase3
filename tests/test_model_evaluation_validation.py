from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from clinical_assistant.acquisition.common import sha256_file
from clinical_assistant.config import project_root
from clinical_assistant.evaluation.validation import (
    validate_evaluation_run,
    validate_stage10_preflight,
)


def _preflight_fixture(root: Path) -> Path:
    source = project_root()
    for relative in ("configs", "data/evaluation", "data/synthetic/hospital"):
        shutil.copytree(source / relative, root / relative)
    for relative in (
        "outputs/baseline/baseline-20260913T143007Z",
        "outputs/training/training-20260913T151805Z",
        "outputs/rag",
    ):
        shutil.copytree(source / relative, root / relative)
    return root


@pytest.mark.integration
def test_stage10_preflight_confirms_real_prerequisites(tmp_path: Path) -> None:
    report = validate_stage10_preflight(_preflight_fixture(tmp_path), write_report=False)
    assert report["ok"] is True
    assert report["variants"] == ["base", "fine_tuned", "fine_tuned_rag"]
    assert report["cases_per_variant"] == 24
    assert report["total_expected_generations"] == 48
    assert report["execution"]["local_gpu_allowed"] is False


@pytest.mark.unit
def test_stage10_preflight_rejects_generation_drift(tmp_path: Path) -> None:
    root = _preflight_fixture(tmp_path)
    path = root / "configs" / "model_evaluation.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["generation"]["temperature"] = 0.1
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    report = validate_stage10_preflight(root, write_report=False)
    assert report["ok"] is False
    assert "generation settings must match the official baseline" in report["errors"]


def _run_fixture(root: Path) -> tuple[Path, Path]:
    (root / "configs").mkdir()
    (root / "data" / "evaluation").mkdir(parents=True)
    output = root / "outputs" / "evaluation"
    run = output / "evaluation-test"
    run.mkdir(parents=True)
    (root / "configs" / "model_evaluation.yaml").write_text(
        yaml.safe_dump(
            {
                "execution": {"expected_cases": 2, "output_directory": "outputs/evaluation"},
            }
        ),
        encoding="utf-8",
    )
    (root / "configs" / "baseline.yaml").write_text(
        yaml.safe_dump({"evaluation": {"required_files": ["cases.jsonl"]}}),
        encoding="utf-8",
    )
    cases = [{"id": "A"}, {"id": "B"}]
    (root / "data" / "evaluation" / "cases.jsonl").write_text(
        "\n".join(json.dumps(item) for item in cases) + "\n", encoding="utf-8"
    )
    hashes = {}
    for variant in ("fine_tuned", "fine_tuned_rag"):
        path = run / f"{variant}_responses.jsonl"
        path.write_text(
            "\n".join(json.dumps({"id": item["id"]}) for item in cases) + "\n",
            encoding="utf-8",
        )
        hashes[variant] = sha256_file(path)
    summary = {
        "stage": 10,
        "complete": True,
        "run_type": "official_full_model_comparison",
        "variants": {
            "base": {},
            "fine_tuned": {"responses_sha256": hashes["fine_tuned"]},
            "fine_tuned_rag": {"responses_sha256": hashes["fine_tuned_rag"]},
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "comparison.md").write_text("# Comparison\n", encoding="utf-8")
    return root, run


@pytest.mark.unit
def test_imported_evaluation_run_validates_hashes_and_cases(tmp_path: Path) -> None:
    root, run = _run_fixture(tmp_path)
    report = validate_evaluation_run(root, run)
    assert report["ok"] is True
    assert report["response_counts"] == {"fine_tuned": 2, "fine_tuned_rag": 2}


@pytest.mark.unit
def test_imported_evaluation_run_detects_tampering(tmp_path: Path) -> None:
    root, run = _run_fixture(tmp_path)
    with (run / "fine_tuned_responses.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "C"}) + "\n")
    report = validate_evaluation_run(root, run)
    assert report["ok"] is False
    assert "fine_tuned response hash mismatch" in report["errors"]
