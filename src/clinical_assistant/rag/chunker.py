"""Deterministic character-aware chunking with source metadata."""

from __future__ import annotations

import hashlib

from clinical_assistant.rag.models import RagChunk, RagDocument


def _split_text(text: str, *, size_chars: int, overlap_chars: int) -> list[str]:
    if size_chars < 100:
        raise ValueError("size_chars must be at least 100")
    if overlap_chars < 0 or overlap_chars >= size_chars:
        raise ValueError("overlap_chars must be between 0 and size_chars - 1")
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        hard_end = min(start + size_chars, len(normalized))
        end = hard_end
        if hard_end < len(normalized):
            boundary = normalized.rfind(" ", start + size_chars // 2, hard_end + 1)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def chunk_documents(
    documents: list[RagDocument],
    *,
    size_chars: int,
    overlap_chars: int,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for document in documents:
        parts = _split_text(
            document.text,
            size_chars=size_chars,
            overlap_chars=overlap_chars,
        )
        for chunk_index, text in enumerate(parts):
            identity = (
                f"{document.document_id}|{document.version}|{document.section}|"
                f"{chunk_index}|{text}"
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            metadata = {
                **document.metadata(),
                "chunk_index": chunk_index,
                "chunk_count": len(parts),
            }
            chunks.append(RagChunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks

