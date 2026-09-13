"""Independent validation for stage 6 chunks, manifest, and Chroma collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path


REQUIRED_METADATA = {
    "document_id",
    "document_name",
    "version",
    "section",
    "source",
    "document_type",
    "synthetic",
    "notice",
    "chunk_index",
    "chunk_count",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rag_index(root: Path, *, check_store: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    config = load_yaml(root / "configs" / "rag.yaml")
    chunks_path = resolve_project_path(config["outputs"]["chunks_file"], root=root)
    manifest_path = resolve_project_path(config["outputs"]["manifest"], root=root)
    chunks: list[dict[str, Any]] = []

    if not chunks_path.is_file():
        errors.append(f"missing chunks file: {chunks_path}")
    else:
        for line_number, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"invalid JSON in chunks line {line_number}")
                continue
            metadata = record.get("metadata")
            if not record.get("chunk_id") or not record.get("text"):
                errors.append(f"incomplete chunk on line {line_number}")
            if not isinstance(metadata, dict) or not REQUIRED_METADATA <= metadata.keys():
                errors.append(f"incomplete metadata on line {line_number}")
            elif metadata.get("synthetic") is not True:
                errors.append(f"chunk not marked synthetic on line {line_number}")
            elif metadata.get("notice") != config["source"]["required_notice"]:
                errors.append(f"invalid academic notice on line {line_number}")
            chunks.append(record)

    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("invalid manifest JSON")
        if manifest and manifest.get("stage") != 6:
            errors.append("manifest stage is not 6")
        if manifest and manifest.get("complete") is not True:
            errors.append("manifest is not complete")
        expected_hash = manifest.get("chunking", {}).get("chunks_sha256")
        if chunks_path.is_file() and expected_hash != _sha256(chunks_path):
            errors.append("chunks SHA-256 does not match manifest")
        if manifest and manifest.get("chunking", {}).get("chunks") != len(chunks):
            errors.append("chunk count does not match manifest")
        source_path = resolve_project_path(config["source"]["protocols_file"], root=root)
        expected_source_hash = manifest.get("source", {}).get("sha256")
        if not source_path.is_file():
            errors.append(f"missing source file: {source_path}")
        elif expected_source_hash != _sha256(source_path):
            errors.append("source SHA-256 does not match manifest")
        embedding_manifest = manifest.get("embeddings", {})
        if embedding_manifest.get("model_id") != config["embeddings"]["model_id"]:
            errors.append("embedding model does not match configuration")
        if embedding_manifest.get("revision") != config["embeddings"]["revision"]:
            errors.append("embedding revision does not match configuration")

    ids = [record.get("chunk_id") for record in chunks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chunk IDs")
    document_ids = sorted(
        {
            str(record.get("metadata", {}).get("document_id"))
            for record in chunks
            if record.get("metadata", {}).get("document_id")
        }
    )
    if len(document_ids) != int(config["source"]["expected_documents"]):
        errors.append("logical document count does not match configuration")

    store_count: int | None = None
    if check_store and manifest:
        try:
            from clinical_assistant.rag.store import ChromaVectorStore

            vector_config = config["vector_store"]
            store = ChromaVectorStore(
                persistence_directory=resolve_project_path(
                    vector_config["persistence_directory"], root=root
                ),
                collection_name=vector_config["collection_name"],
                distance=vector_config["distance"],
            )
            store_count = store.count()
            if store_count != len(chunks):
                errors.append("Chroma collection count does not match chunks")
        except Exception as exc:
            errors.append(f"could not validate Chroma collection: {exc}")

    return {
        "ok": not errors,
        "stage": 6,
        "chunks": len(chunks),
        "logical_documents": len(document_ids),
        "document_ids": document_ids,
        "store_count": store_count,
        "errors": errors,
    }
