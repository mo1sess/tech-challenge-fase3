"""Structural curation and exact-deduplication for instruction examples."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from clinical_assistant.preprocessing.anonymizer import anonymize_text


TEXT_FIELDS = ("instruction", "input", "output")


def content_hash(example: dict[str, Any]) -> str:
    """Return a stable fingerprint used for deduplication and split assignment."""

    canonical = "\x1f".join(
        [str(example.get("category", "")).casefold()]
        + [str(example.get(field, "")).casefold() for field in TEXT_FIELDS]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def curate_examples(
    examples: Iterable[dict[str, Any]],
    *,
    min_instruction_chars: int = 8,
    min_output_chars: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clean, anonymize, reject incomplete rows, and remove exact duplicates."""

    stats = {
        "seen": 0,
        "accepted": 0,
        "rejected_short_instruction": 0,
        "rejected_short_output": 0,
        "rejected_duplicate": 0,
    }
    curated: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for original in examples:
        stats["seen"] += 1
        item = dict(original)
        for field in TEXT_FIELDS:
            item[field] = anonymize_text(item.get(field, ""))
        if len(item["instruction"]) < min_instruction_chars:
            stats["rejected_short_instruction"] += 1
            continue
        if len(item["output"]) < min_output_chars:
            stats["rejected_short_output"] += 1
            continue
        fingerprint = content_hash(item)
        if fingerprint in fingerprints:
            stats["rejected_duplicate"] += 1
            continue
        fingerprints.add(fingerprint)
        item["content_hash"] = fingerprint
        curated.append(item)
    stats["accepted"] = len(curated)
    return curated, stats

