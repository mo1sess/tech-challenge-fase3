from __future__ import annotations

from pathlib import Path

import pytest

from clinical_assistant.acquisition.protocols import import_protocols


@pytest.mark.unit
def test_protocol_import_allows_only_document_types(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    target = tmp_path / "raw"
    inbox.mkdir()
    (inbox / "ASM-TEST.md").write_text(
        "DOCUMENTO SINTETICO PARA FINS ACADEMICOS", encoding="utf-8"
    )
    (inbox / "malware.exe").write_bytes(b"no")
    result = import_protocols(inbox, target, allowed_extensions={".pdf", ".txt", ".md"})
    assert result["counts"] == {"imported_files": 1, "rejected_files": 1}
    assert (target / "ASM-TEST.md").exists()
    assert not (target / "malware.exe").exists()

