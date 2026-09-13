"""Prepare a deterministic, leakage-resistant dataset for QLoRA SFT."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.evaluation.dataset import normalize_for_comparison
from clinical_assistant.preprocessing.dataset_builder import write_jsonl


REQUIRED_FIELDS = {
    "category",
    "content_hash",
    "id",
    "input",
    "instruction",
    "metadata",
    "notice",
    "output",
    "source",
    "synthetic",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: expected an object")
            records.append(record)
    return records


def _validate_source_record(record: dict[str, Any], *, notice: str) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"{record.get('id', '<unknown>')}: missing fields {sorted(missing)}")
    if record["synthetic"] is not True:
        raise ValueError(f"{record['id']}: stage 5 source must be explicitly synthetic")
    if record["notice"] != notice:
        raise ValueError(f"{record['id']}: missing synthetic academic notice")
    for field in ("id", "category", "content_hash", "instruction", "output", "source"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{record.get('id', '<unknown>')}: invalid {field}")
    if not isinstance(record["input"], str) or not isinstance(record["metadata"], dict):
        raise ValueError(f"{record['id']}: invalid input or metadata")


def load_synthetic_training_records(
    directory: Path, required_files: list[str], *, notice: str
) -> list[dict[str, Any]]:
    """Load and validate every configured Hospital TechCare record."""

    records: list[dict[str, Any]] = []
    for filename in required_files:
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"Missing synthetic training file: {filename}")
        for record in _load_jsonl(path):
            _validate_source_record(record, notice=notice)
            records.append(record)
    ids = [str(record["id"]) for record in records]
    hashes = [str(record["content_hash"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate synthetic training ID")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate synthetic training content hash")
    return records


def split_records_by_category(
    records: list[dict[str, Any]],
    *,
    train_percent: int,
    validation_percent: int,
    test_percent: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic stratified splits with every category represented."""

    if train_percent + validation_percent + test_percent != 100:
        raise ValueError("Split percentages must sum to 100")
    if min(train_percent, validation_percent, test_percent) <= 0:
        raise ValueError("Every split percentage must be positive")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for category, values in sorted(grouped.items()):
        if len(values) < 3:
            raise ValueError(f"Category {category} needs at least three records")
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item['content_hash']}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count = max(1, len(ordered) * validation_percent // 100)
        test_count = max(1, len(ordered) * test_percent // 100)
        train_count = len(ordered) - validation_count - test_count
        if train_count < 1:
            raise ValueError(f"Category {category} has no room for the training split")
        splits["train"].extend(ordered[:train_count])
        splits["validation"].extend(ordered[train_count : train_count + validation_count])
        splits["test"].extend(ordered[train_count + validation_count :])

    for split_records in splits.values():
        split_records.sort(key=lambda item: (str(item["category"]), str(item["content_hash"])))
    return splits


def _user_message(record: dict[str, Any]) -> str:
    instruction = str(record["instruction"]).strip()
    input_text = str(record["input"]).strip()
    return f"{instruction}\n\nContexto:\n{input_text}" if input_text else instruction


def to_conversational_record(record: dict[str, Any], *, system_prompt: str) -> dict[str, Any]:
    """Convert an internal example to TRL conversational prompt-completion format."""

    return {
        "id": record["id"],
        "category": record["category"],
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_message(record)},
        ],
        "completion": [{"role": "assistant", "content": str(record["output"]).strip()}],
        "source": record["source"],
        "synthetic": True,
        "content_hash": record["content_hash"],
        "metadata": record["metadata"],
    }


def _evaluation_questions(root: Path, directory: str, filenames: list[str]) -> set[str]:
    questions: set[str] = set()
    base = resolve_project_path(directory, root=root)
    for filename in filenames:
        for record in _load_jsonl(base / filename):
            question = record.get("question")
            if isinstance(question, str) and question.strip():
                questions.add(normalize_for_comparison(question))
    return questions


def _category_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["category"]) for record in records).items()))


def prepare_finetuning_dataset(root: Path) -> dict[str, Any]:
    """Build stage 5 train/validation/test files and an evidence manifest."""

    config_path = root / "configs" / "training.yaml"
    config = load_yaml(config_path)
    if int(config.get("stage", -1)) != 5:
        raise ValueError("Training configuration must declare stage 5")
    dataset_config = config["dataset"]
    split_config = dataset_config["split"]
    records = load_synthetic_training_records(
        resolve_project_path(dataset_config["source_directory"], root=root),
        list(map(str, dataset_config["required_files"])),
        notice=str(dataset_config["required_notice"]),
    )
    expected = int(dataset_config["expected_source_records"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} source records, found {len(records)}")

    splits = split_records_by_category(
        records,
        train_percent=int(split_config["train_percent"]),
        validation_percent=int(split_config["validation_percent"]),
        test_percent=int(split_config["test_percent"]),
        seed=int(config["reproducibility"]["seed"]),
    )
    evaluation_questions = _evaluation_questions(
        root,
        str(dataset_config["evaluation_directory"]),
        list(map(str, dataset_config["evaluation_files"])),
    )
    source_questions = {
        normalize_for_comparison(str(record["instruction"])) for record in records
    }
    overlap = sorted(source_questions & evaluation_questions)
    if overlap:
        raise ValueError(f"Exact leakage into the evaluation dataset: {len(overlap)} questions")

    output_directory = resolve_project_path(dataset_config["output_directory"], root=root)
    output_directory.mkdir(parents=True, exist_ok=True)
    file_details: dict[str, dict[str, Any]] = {}
    split_hashes: dict[str, set[str]] = {}
    for name, split_records in splits.items():
        conversational = [
            to_conversational_record(record, system_prompt=str(dataset_config["system_prompt"]))
            for record in split_records
        ]
        path = output_directory / f"{name}.jsonl"
        write_jsonl(path, conversational)
        split_hashes[name] = {str(record["content_hash"]) for record in conversational}
        file_details[path.name] = {
            "records": len(conversational),
            "categories": _category_counts(conversational),
            "sha256": sha256_file(path),
        }

    if split_hashes["train"] & split_hashes["validation"]:
        raise ValueError("Data leakage between train and validation")
    if split_hashes["train"] & split_hashes["test"]:
        raise ValueError("Data leakage between train and test")
    if split_hashes["validation"] & split_hashes["test"]:
        raise ValueError("Data leakage between validation and test")

    manifest = {
        "schema_version": 1,
        "stage": 5,
        "generated_at_utc": utc_now(),
        "source": "Hospital TechCare (hospital fictício)",
        "synthetic": True,
        "source_records": len(records),
        "files": file_details,
        "leakage_check": {
            "ok": True,
            "exact_evaluation_question_overlap": 0,
            "shared_content_hashes_between_splits": 0,
        },
        "configuration_sha256": sha256_file(config_path),
        "limitations": [
            "The dataset is synthetic academic material, not real hospital data.",
            "Automatic structural validation is not clinical review.",
            "The small dataset is intended to teach behavior and response structure.",
        ],
    }
    write_json(resolve_project_path(dataset_config["manifest"], root=root), manifest)
    return manifest


def validate_finetuning_dataset(root: Path) -> dict[str, Any]:
    """Validate generated stage 5 files against their manifest and schema."""

    config = load_yaml(root / "configs" / "training.yaml")
    dataset_config = config["dataset"]
    manifest_path = resolve_project_path(dataset_config["manifest"], root=root)
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"ok": False, "errors": ["Missing fine-tuning dataset manifest"]}
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    output_directory = resolve_project_path(dataset_config["output_directory"], root=root)
    observed_hashes: set[str] = set()
    counts: dict[str, int] = {}
    categories: dict[str, dict[str, int]] = {}
    for filename, details in manifest.get("files", {}).items():
        path = output_directory / filename
        if not path.is_file():
            errors.append(f"Missing generated split: {filename}")
            continue
        records = _load_jsonl(path)
        counts[path.stem] = len(records)
        categories[path.stem] = _category_counts(records)
        if len(records) != int(details["records"]):
            errors.append(f"Record count mismatch: {filename}")
        if sha256_file(path) != details["sha256"]:
            errors.append(f"SHA-256 mismatch: {filename}")
        for record in records:
            if not isinstance(record.get("prompt"), list) or not isinstance(
                record.get("completion"), list
            ):
                errors.append(f"Invalid conversational schema: {filename}")
                break
            fingerprint = str(record.get("content_hash", ""))
            if not fingerprint or fingerprint in observed_hashes:
                errors.append(f"Duplicate or missing content hash: {filename}")
            observed_hashes.add(fingerprint)
    required_splits = {"train", "validation", "test"}
    if set(counts) != required_splits:
        errors.append("Generated dataset must contain train, validation and test splits")
    if sum(counts.values()) != int(manifest.get("source_records", -1)):
        errors.append("Generated total differs from source total")
    return {
        "ok": not errors,
        "stage": 5,
        "counts": counts,
        "categories": categories,
        "total_records": sum(counts.values()),
        "unique_content_hashes": len(observed_hashes),
        "errors": errors,
    }
