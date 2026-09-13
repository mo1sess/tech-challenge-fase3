"""Loading and integrity checks for the held-out evaluation suite."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path


REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "context",
    "required_any_groups",
    "requires_refusal",
    "requires_human_validation",
    "expects_source_limitation",
}
EXPECTED_CATEGORIES = {"clinical", "protocol", "patient", "safety"}


def normalize_for_comparison(value: str) -> str:
    """Normalize text for conservative exact leakage checks."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).split())


def load_evaluation_cases(directory: Path, required_files: list[str]) -> list[dict[str, Any]]:
    """Load the configured JSONL files in stable order."""

    cases: list[dict[str, Any]] = []
    for filename in required_files:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation file: {path}")
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{filename}:{line_number}: invalid JSON") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{filename}:{line_number}: expected object")
                cases.append(item)
    return cases


def _training_instructions(root: Path) -> tuple[set[str], list[str]]:
    instructions: set[str] = set()
    checked: list[str] = []
    patterns = (
        root / "data" / "processed" / "training" / "*.jsonl",
        root / "data" / "synthetic" / "hospital" / "*.jsonl",
    )
    for pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            if path.name.endswith("manifest.json"):
                continue
            checked.append(path.relative_to(root).as_posix())
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    instruction = item.get("instruction")
                    if instruction:
                        instructions.add(normalize_for_comparison(str(instruction)))
    return instructions, checked


def validate_evaluation_dataset(root: Path) -> dict[str, Any]:
    """Validate schema, categories, uniqueness, and exact train/eval leakage."""

    config = load_yaml(root / "configs" / "baseline.yaml")
    eval_config = config["evaluation"]
    directory = resolve_project_path(config["execution"]["evaluation_directory"], root=root)
    required_files = list(map(str, eval_config["required_files"]))
    errors: list[str] = []
    ids: set[str] = set()
    questions: set[str] = set()
    counts: dict[str, int] = {category: 0 for category in sorted(EXPECTED_CATEGORIES)}
    file_hashes = {
        filename: sha256_file(directory / filename)
        for filename in required_files
        if (directory / filename).exists()
    }

    try:
        cases = load_evaluation_cases(directory, required_files)
    except (FileNotFoundError, ValueError) as exc:
        cases = []
        errors.append(str(exc))

    for index, item in enumerate(cases, 1):
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            errors.append(f"record {index}: missing fields {sorted(missing)}")
            continue
        identifier = str(item["id"])
        normalized_question = normalize_for_comparison(str(item["question"]))
        category = str(item["category"])
        if identifier in ids:
            errors.append(f"record {index}: duplicate id {identifier}")
        if normalized_question in questions:
            errors.append(f"record {index}: duplicate question")
        if category not in EXPECTED_CATEGORIES:
            errors.append(f"record {index}: unexpected category {category}")
        else:
            counts[category] += 1
        groups = item["required_any_groups"]
        if not isinstance(groups, list) or any(not isinstance(group, list) or not group for group in groups):
            errors.append(f"record {index}: invalid required_any_groups")
        ids.add(identifier)
        questions.add(normalized_question)

    expected_count = int(eval_config["expected_cases"])
    if len(cases) != expected_count:
        errors.append(f"expected {expected_count} cases, found {len(cases)}")

    training, checked_files = _training_instructions(root)
    overlaps = sorted(questions & training)
    if overlaps:
        errors.append(f"exact train/eval leakage detected in {len(overlaps)} questions")

    report = {
        "validated_at_utc": utc_now(),
        "stage": 4,
        "ok": not errors,
        "total_cases": len(cases),
        "counts": counts,
        "unique_ids": len(ids),
        "unique_questions": len(questions),
        "exact_leakage_count": len(overlaps),
        "training_files_checked": checked_files,
        "file_sha256": file_hashes,
        "errors": errors,
    }
    write_json(root / "outputs" / "evaluation_data_validation.json", report)
    return report
