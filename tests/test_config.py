from __future__ import annotations

from pathlib import Path

import pytest

from clinical_assistant.config import load_yaml, resolve_project_path


@pytest.mark.unit
def test_dataset_destinations_are_relative_and_scoped() -> None:
    root = Path(__file__).resolve().parents[1]
    datasets = load_yaml(root / "configs" / "datasets.yaml")["datasets"]
    for item in datasets.values():
        for field in ("destination", "inbox"):
            if field in item:
                configured = Path(item[field])
                assert not configured.is_absolute()
                assert resolve_project_path(configured, root=root).is_relative_to(root)


@pytest.mark.unit
def test_resolve_project_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes project root"):
        resolve_project_path("../outside", root=tmp_path / "project")

