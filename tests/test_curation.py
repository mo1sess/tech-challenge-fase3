from __future__ import annotations

import pytest

from clinical_assistant.preprocessing.curation import curate_examples
from clinical_assistant.preprocessing.dataset_builder import assign_split, split_examples


def _example(instruction: str, output: str, *, source: str = "fixture") -> dict[str, object]:
    return {
        "category": "medical_qa",
        "instruction": instruction,
        "input": "",
        "output": output,
        "source": source,
        "synthetic": False,
        "metadata": {},
    }


@pytest.mark.unit
def test_curation_rejects_empty_and_duplicate_examples() -> None:
    examples = [
        _example("What is asthma?", "A chronic respiratory condition."),
        _example(" What  is asthma? ", "A chronic respiratory condition."),
        _example("Valid question?", ""),
        _example("x", "Long enough answer"),
    ]
    curated, stats = curate_examples(examples)
    assert len(curated) == 1
    assert stats == {
        "seen": 4,
        "accepted": 1,
        "rejected_short_instruction": 1,
        "rejected_short_output": 1,
        "rejected_duplicate": 1,
    }


@pytest.mark.unit
def test_split_is_deterministic_and_has_no_leakage() -> None:
    curated, _ = curate_examples(
        [_example(f"Question number {index}", f"Answer number {index}") for index in range(200)]
    )
    percentages = {"train": 80, "validation": 10, "test": 10}
    first = split_examples(curated, percentages)
    second = split_examples(reversed(curated), percentages)
    assert {
        name: [item["content_hash"] for item in records] for name, records in first.items()
    } == {
        name: [item["content_hash"] for item in records] for name, records in second.items()
    }
    hashes = [{item["content_hash"] for item in first[name]} for name in first]
    assert hashes[0].isdisjoint(hashes[1])
    assert hashes[0].isdisjoint(hashes[2])
    assert hashes[1].isdisjoint(hashes[2])


@pytest.mark.unit
def test_invalid_split_percentages_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 100"):
        assign_split("0" * 64, {"train": 80, "validation": 10, "test": 5})

