"""Build deterministic, leakage-resistant instruction datasets."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.preprocessing.anonymizer import (
    SAFE_PATIENT_COLUMNS,
    anonymize_synthea_directory,
)
from clinical_assistant.preprocessing.cleaner import normalize_text
from clinical_assistant.preprocessing.curation import curate_examples


def _element_text(element: ET.Element | None) -> str:
    return normalize_text("".join(element.itertext()) if element is not None else "")


def iter_medquad(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield raw MedQuAD examples with provenance from every XML file."""

    for path in sorted(directory.rglob("*.xml")):
        document = ET.parse(path).getroot()
        source = document.attrib.get("source", "MedQuAD")
        url = document.attrib.get("url", "")
        focus = normalize_text(document.findtext("Focus", default=""))
        for pair in document.findall("./QAPairs/QAPair"):
            question = pair.find("Question")
            answer = pair.find("Answer")
            yield {
                "category": "medical_qa",
                "instruction": _element_text(question),
                "input": "",
                "output": _element_text(answer),
                "source": "MedQuAD",
                "synthetic": False,
                "metadata": {
                    "document_source": source,
                    "document_url": url,
                    "focus": focus,
                    "question_id": question.attrib.get("qid", "") if question is not None else "",
                    "question_type": question.attrib.get("qtype", "") if question is not None else "",
                    "raw_file": path.relative_to(directory).as_posix(),
                },
            }


def iter_pubmedqa(path: Path) -> Iterator[dict[str, Any]]:
    """Yield PQA-L examples, preserving evidence and the labeled decision."""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("PubMedQA PQA-L must be keyed by PMID")
    for pmid in sorted(payload):
        record = payload[pmid]
        contexts = record.get("CONTEXTS") or []
        evidence = "\n".join(
            f"Evidence {index}: {normalize_text(context)}"
            for index, context in enumerate(contexts, 1)
        )
        decision = normalize_text(record.get("final_decision", ""))
        long_answer = normalize_text(record.get("LONG_ANSWER", ""))
        answer = f"Decision: {decision}. Explanation: {long_answer}" if decision else long_answer
        yield {
            "category": "biomedical_evidence_qa",
            "instruction": normalize_text(record.get("QUESTION", "")),
            "input": evidence,
            "output": answer,
            "source": "PubMedQA/PQA-L",
            "synthetic": False,
            "metadata": {
                "pmid": str(pmid),
                "year": normalize_text(record.get("YEAR", "")),
                "meshes": [normalize_text(value) for value in record.get("MESHES", [])],
                "label": decision,
            },
        }


def assign_split(content_fingerprint: str, percentages: dict[str, int]) -> str:
    """Assign a stable split from content only, preventing duplicate leakage."""

    if sum(percentages.values()) != 100 or any(value < 0 for value in percentages.values()):
        raise ValueError("Split percentages must be non-negative and sum to 100")
    bucket = int(content_fingerprint[:8], 16) % 100
    train_end = percentages["train"]
    validation_end = train_end + percentages["validation"]
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def split_examples(
    examples: Iterable[dict[str, Any]], percentages: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Split curated examples and verify that no fingerprint crosses partitions."""

    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[str] = set()
    for example in examples:
        fingerprint = str(example["content_hash"])
        if fingerprint in seen:
            raise ValueError(f"Duplicate content hash before split: {fingerprint}")
        seen.add(fingerprint)
        partitions[assign_split(fingerprint, percentages)].append(example)
    for values in partitions.values():
        values.sort(key=lambda item: str(item["content_hash"]))
    return partitions


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write one UTF-8 JSON object per line and return the record count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _source_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["source"]) for record in records).items()))


def build_processed_datasets(root: Path) -> dict[str, Any]:
    """Run all stage 2 processing and persist outputs plus an audit manifest."""

    config = load_yaml(root / "configs" / "preprocessing.yaml")
    text_config = config["text"]
    split_config = config["split"]
    output_config = config["outputs"]
    percentages = {
        "train": int(split_config["train_percent"]),
        "validation": int(split_config["validation_percent"]),
        "test": int(split_config["test_percent"]),
    }
    configured_patient_columns = tuple(config["synthea"]["safe_patient_columns"])
    if configured_patient_columns != SAFE_PATIENT_COLUMNS:
        raise ValueError(
            "Configured Synthea patient schema differs from the approved safe schema"
        )

    medquad_dir = root / "data" / "raw" / "medquad"
    pubmedqa_matches = list((root / "data" / "raw" / "pubmedqa").rglob("ori_pqal.json"))
    if len(pubmedqa_matches) != 1:
        raise ValueError(f"Expected one PubMedQA PQA-L file, found {len(pubmedqa_matches)}")

    raw_examples = list(iter_medquad(medquad_dir))
    medquad_seen = len(raw_examples)
    raw_examples.extend(iter_pubmedqa(pubmedqa_matches[0]))
    curated, curation_stats = curate_examples(
        raw_examples,
        min_instruction_chars=int(text_config["min_instruction_chars"]),
        min_output_chars=int(text_config["min_output_chars"]),
    )
    partitions = split_examples(curated, percentages)

    training_dir = resolve_project_path(output_config["training_directory"], root=root)
    split_details: dict[str, dict[str, Any]] = {}
    split_hashes: dict[str, set[str]] = {}
    for name, records in partitions.items():
        count = write_jsonl(training_dir / f"{name}.jsonl", records)
        split_hashes[name] = {str(record["content_hash"]) for record in records}
        split_details[name] = {"records": count, "sources": _source_counts(records)}

    if split_hashes["train"] & split_hashes["validation"]:
        raise ValueError("Data leakage between train and validation")
    if split_hashes["train"] & split_hashes["test"]:
        raise ValueError("Data leakage between train and test")
    if split_hashes["validation"] & split_hashes["test"]:
        raise ValueError("Data leakage between validation and test")

    synthea_result = anonymize_synthea_directory(
        root / "data" / "raw" / "synthea",
        resolve_project_path(output_config["synthea_directory"], root=root),
        prefix=str(config["synthea"]["patient_prefix"]),
    )

    raw_manifests: dict[str, Any] = {}
    for manifest_path in sorted((root / "data" / "raw" / "_manifests").glob("*.json")):
        with manifest_path.open("r", encoding="utf-8") as stream:
            raw_manifests[manifest_path.stem] = json.load(stream)

    manifest = {
        "schema_version": 1,
        "stage": 2,
        "generated_at_utc": utc_now(),
        "configuration": config,
        "raw_counts": {
            "medquad_examples": medquad_seen,
            "pubmedqa_examples": len(raw_examples) - medquad_seen,
        },
        "curation": curation_stats,
        "splits": split_details,
        "leakage_check": {"ok": True, "shared_content_hashes": 0},
        "synthea_anonymization": synthea_result,
        "raw_manifests": raw_manifests,
        "limitations": [
            "Structural curation does not constitute clinical validation.",
            "MedQuAD and PubMedQA content is predominantly English.",
            "No local clinical protocols were available in this stage.",
            "Synthea records are synthetic but were anonymized to demonstrate the control.",
        ],
    }
    write_json(resolve_project_path(output_config["manifest"], root=root), manifest)
    return manifest
