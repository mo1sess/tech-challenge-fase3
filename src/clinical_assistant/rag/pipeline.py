"""Build and open the stage 6 local RAG index."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.rag.chunker import chunk_documents
from clinical_assistant.rag.embeddings import SentenceTransformerEmbeddingProvider
from clinical_assistant.rag.loader import load_synthetic_protocols
from clinical_assistant.rag.retriever import ProtocolRetriever
from clinical_assistant.rag.store import ChromaVectorStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values
    )
    path.write_text(payload + "\n", encoding="utf-8")


def load_rag_config(root: Path) -> dict[str, Any]:
    return load_yaml(root / "configs" / "rag.yaml")


def _components(root: Path, config: dict[str, Any]):
    embeddings_config = config["embeddings"]
    vector_config = config["vector_store"]
    provider = SentenceTransformerEmbeddingProvider(
        model_id=embeddings_config["model_id"],
        revision=embeddings_config["revision"],
        device=embeddings_config["device"],
    )
    store = ChromaVectorStore(
        persistence_directory=resolve_project_path(
            vector_config["persistence_directory"], root=root
        ),
        collection_name=vector_config["collection_name"],
        distance=vector_config["distance"],
    )
    return provider, store


def build_rag_index(root: Path) -> dict[str, Any]:
    config = load_rag_config(root)
    source_config = config["source"]
    source_path = resolve_project_path(source_config["protocols_file"], root=root)
    documents = load_synthetic_protocols(
        source_path,
        required_notice=source_config["required_notice"],
        expected_records=int(source_config["expected_records"]),
    )
    expected_documents = int(source_config["expected_documents"])
    if len(documents) != expected_documents:
        raise ValueError(f"Expected {expected_documents} documents, found {len(documents)}")

    chunk_config = config["chunking"]
    chunks = chunk_documents(
        documents,
        size_chars=int(chunk_config["size_chars"]),
        overlap_chars=int(chunk_config["overlap_chars"]),
    )
    chunks_path = resolve_project_path(config["outputs"]["chunks_file"], root=root)
    _write_jsonl(chunks_path, [chunk.to_dict() for chunk in chunks])

    provider, store = _components(root, config)
    embeddings = provider.encode([chunk.text for chunk in chunks])
    indexed_count = store.rebuild(chunks, embeddings)
    vector_config = config["vector_store"]
    manifest = {
        "schema_version": 1,
        "stage": 6,
        "complete": indexed_count == len(chunks),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": source_config["protocols_file"],
            "kind": source_config["kind"],
            "sha256": _sha256(source_path),
            "records": int(source_config["expected_records"]),
            "logical_documents": len(documents),
            "synthetic": True,
            "notice": source_config["required_notice"],
        },
        "chunking": {
            "size_chars": int(chunk_config["size_chars"]),
            "overlap_chars": int(chunk_config["overlap_chars"]),
            "chunks": len(chunks),
            "chunks_sha256": _sha256(chunks_path),
        },
        "embeddings": {
            "provider": config["embeddings"]["provider"],
            "model_id": provider.model_id,
            "revision": provider.revision,
            "device": "cpu",
            "dimensions": len(embeddings[0]),
            "normalized": True,
        },
        "vector_store": {
            "provider": vector_config["provider"],
            "collection_name": vector_config["collection_name"],
            "persistence_directory": vector_config["persistence_directory"],
            "distance": vector_config["distance"],
            "indexed_chunks": indexed_count,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "chromadb": importlib.metadata.version("chromadb"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "gpu_used": False,
        },
        "limitations": [
            "A base contém apenas protocolos internos sintéticos para demonstração acadêmica.",
            "Os resultados recuperados não estabelecem correção clínica nem autorizam conduta.",
            "Toda decisão clínica exige validação humana e uma fonte clínica oficial revisada.",
        ],
    }
    manifest_path = resolve_project_path(config["outputs"]["manifest"], root=root)
    _write_json(manifest_path, manifest)
    return manifest


def open_protocol_retriever(root: Path) -> ProtocolRetriever:
    config = load_rag_config(root)
    provider, store = _components(root, config)
    retrieval = config["retrieval"]
    return ProtocolRetriever(
        embedding_provider=provider,
        vector_store=store,
        top_k=int(retrieval["top_k"]),
        minimum_relevance=float(retrieval["minimum_relevance"]),
    )
