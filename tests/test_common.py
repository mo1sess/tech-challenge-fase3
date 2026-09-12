from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from clinical_assistant.acquisition.common import safe_extract_zip, sha256_file


@pytest.mark.unit
def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"clinical-data")
    assert sha256_file(path) == "5490c06140bd564bc3e5e699899bf39a9d8301075024ff3556093618a4c37968"


@pytest.mark.unit
def test_safe_extract_zip_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "blocked")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "out")
