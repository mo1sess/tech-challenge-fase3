from __future__ import annotations

from pathlib import Path

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.rag.chunker import chunk_documents
from clinical_assistant.rag.embeddings import DeterministicHashEmbeddingProvider
from clinical_assistant.rag.loader import load_synthetic_protocols
from clinical_assistant.rag.retriever import ProtocolRetriever
from clinical_assistant.rag.store import ChromaVectorStore


NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


@pytest.mark.integration
def test_chroma_persists_and_returns_source_aware_hits(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    documents = load_synthetic_protocols(
        project_root() / "data" / "synthetic" / "hospital" / "protocols.jsonl",
        required_notice=NOTICE,
        expected_records=15,
    )
    chunks = chunk_documents(documents, size_chars=1200, overlap_chars=150)
    provider = DeterministicHashEmbeddingProvider()
    store = ChromaVectorStore(
        persistence_directory=tmp_path / "chroma",
        collection_name="test_protocols",
    )
    assert store.rebuild(chunks, provider.encode([chunk.text for chunk in chunks])) == len(chunks)

    reopened = ChromaVectorStore(
        persistence_directory=tmp_path / "chroma",
        collection_name="test_protocols",
    )
    assert reopened.count() == len(chunks)
    retriever = ProtocolRetriever(
        embedding_provider=provider,
        vector_store=reopened,
        top_k=3,
        minimum_relevance=0.0,
    )
    hits = retriever.retrieve("Como verificar exames pendentes?")
    assert hits
    assert all(hit.metadata["document_id"].startswith("ASM-") for hit in hits)
    assert all("fonte:" in hit.citation for hit in hits)

