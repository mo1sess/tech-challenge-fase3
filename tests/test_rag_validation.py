from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.rag.chunker import chunk_documents
from clinical_assistant.rag.loader import load_synthetic_protocols
from clinical_assistant.validation.rag import validate_rag_index


NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


def _prepared_root(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root() / "configs" / "rag.yaml", tmp_path / "configs" / "rag.yaml")
    source = tmp_path / "data" / "synthetic" / "hospital" / "protocols.jsonl"
    source.parent.mkdir(parents=True)
    shutil.copy(
        project_root() / "data" / "synthetic" / "hospital" / "protocols.jsonl",
        source,
    )
    documents = load_synthetic_protocols(
        source, required_notice=NOTICE, expected_records=15
    )
    chunks = chunk_documents(documents, size_chars=1200, overlap_chars=150)
    chunks_path = tmp_path / "data" / "processed" / "rag" / "protocol_chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text(
        "\n".join(
            json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True)
            for chunk in chunks
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "outputs" / "rag" / "index_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "stage": 6,
        "complete": True,
        "source": {
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "chunking": {
            "chunks": len(chunks),
            "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        },
        "embeddings": {
            "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.mark.integration
def test_rag_artifacts_pass_independent_validation(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)
    report = validate_rag_index(root, check_store=False)
    assert report["ok"] is True
    assert report["logical_documents"] == 5
    assert report["document_ids"] == ["ASM-001", "ASM-002", "ASM-003", "ASM-004", "ASM-005"]


@pytest.mark.unit
def test_rag_validator_detects_chunk_tampering(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)
    chunks_path = root / "data" / "processed" / "rag" / "protocol_chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    report = validate_rag_index(root, check_store=False)
    assert report["ok"] is False
    assert "chunks SHA-256 does not match manifest" in report["errors"]
