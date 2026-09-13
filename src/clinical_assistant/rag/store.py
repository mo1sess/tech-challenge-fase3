"""Thin ChromaDB adapter with explicit embeddings and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clinical_assistant.rag.models import RagChunk, RetrievalHit


def _chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed; install requirements/rag-local.txt"
        ) from exc
    return chromadb


def _chroma_settings():
    from chromadb.config import Settings

    return Settings(anonymized_telemetry=False)


class ChromaVectorStore:
    def __init__(
        self,
        *,
        persistence_directory: Path,
        collection_name: str,
        distance: str = "cosine",
    ) -> None:
        persistence_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance = distance
        self.client = _chromadb().PersistentClient(
            path=str(persistence_directory),
            settings=_chroma_settings(),
        )

    def rebuild(self, chunks: list[RagChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match")
        if not chunks:
            raise ValueError("Cannot build an empty RAG collection")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise
        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )
        return collection.count()

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).count()

    def query(self, embedding: list[float], *, top_k: int) -> list[RetrievalHit]:
        collection = self.client.get_collection(self.collection_name)
        available = collection.count()
        if available == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[RetrievalHit] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            numeric_distance = float(distance)
            hits.append(
                RetrievalHit(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    metadata=dict(metadata or {}),
                    distance=numeric_distance,
                    relevance=max(0.0, min(1.0, 1.0 - numeric_distance)),
                )
            )
        return hits
