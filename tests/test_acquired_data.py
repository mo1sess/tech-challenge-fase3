from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from clinical_assistant.acquisition.datasets import COUNTERS


@pytest.mark.unit
def test_supported_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.integration
@pytest.mark.parametrize("dataset", ["medquad", "pubmedqa", "synthea"])
def test_acquired_counts_match_manifests(dataset: str) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / dataset
    payload = [path for path in raw.iterdir() if path.name != ".gitkeep"]
    if not payload:
        pytest.skip("raw data is intentionally not stored in Git; run acquire_data.py")
    with (root / "data" / "raw" / "_manifests" / f"{dataset}.json").open(
        "r", encoding="utf-8"
    ) as stream:
        manifest = json.load(stream)
    assert COUNTERS[dataset](raw) == manifest["counts"]

