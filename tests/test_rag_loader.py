from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.rag.chunker import chunk_documents
from clinical_assistant.rag.loader import load_synthetic_protocols
from clinical_assistant.rag.models import RetrievalHit


NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


@pytest.mark.unit
def test_loader_deduplicates_prompt_variants_into_five_documents() -> None:
    path = project_root() / "data" / "synthetic" / "hospital" / "protocols.jsonl"
    documents = load_synthetic_protocols(
        path, required_notice=NOTICE, expected_records=15
    )
    assert [document.document_id for document in documents] == [
        "ASM-001",
        "ASM-002",
        "ASM-003",
        "ASM-004",
        "ASM-005",
    ]
    assert all(document.synthetic for document in documents)
    assert all(document.notice == NOTICE for document in documents)


@pytest.mark.unit
def test_loader_rejects_unlabelled_synthetic_material(tmp_path: Path) -> None:
    source = project_root() / "data" / "synthetic" / "hospital" / "protocols.jsonl"
    records = source.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["notice"] = ""
    records[0] = json.dumps(first, ensure_ascii=False)
    path = tmp_path / "protocols.jsonl"
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="academic notice"):
        load_synthetic_protocols(path, required_notice=NOTICE, expected_records=15)


@pytest.mark.unit
def test_chunks_have_stable_ids_and_complete_citation_metadata() -> None:
    path = project_root() / "data" / "synthetic" / "hospital" / "protocols.jsonl"
    documents = load_synthetic_protocols(
        path, required_notice=NOTICE, expected_records=15
    )
    first = chunk_documents(documents, size_chars=1200, overlap_chars=150)
    second = chunk_documents(documents, size_chars=1200, overlap_chars=150)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) >= 5
    required = {
        "document_id",
        "document_name",
        "version",
        "section",
        "source",
        "document_type",
        "synthetic",
        "notice",
    }
    assert all(required <= chunk.metadata.keys() for chunk in first)


@pytest.mark.unit
def test_retrieval_hit_formats_explainable_citation() -> None:
    hit = RetrievalHit(
        chunk_id="chunk-1",
        text="conteúdo",
        metadata={
            "document_name": "Protocolo Interno Sintético ASM-002",
            "version": "1.0",
            "section": "2.1",
            "source": "Hospital TechCare (hospital fictício)",
        },
        distance=0.1,
        relevance=0.9,
    )
    assert "ASM-002" in hit.citation
    assert "versão 1.0" in hit.citation
    assert "seção 2.1" in hit.citation
    assert "hospital fictício" in hit.citation

