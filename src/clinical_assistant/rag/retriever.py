"""Protocol retrieval with minimum relevance and formatted citations."""

from __future__ import annotations

from clinical_assistant.rag.embeddings import EmbeddingProvider
from clinical_assistant.rag.models import RetrievalHit
from clinical_assistant.rag.store import ChromaVectorStore


class ProtocolRetriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        top_k: int = 3,
        minimum_relevance: float = 0.20,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= minimum_relevance <= 1.0:
            raise ValueError("minimum_relevance must be between 0 and 1")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.minimum_relevance = minimum_relevance

    def retrieve(self, query: str) -> list[RetrievalHit]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Query cannot be empty")
        embedding = self.embedding_provider.encode([normalized_query])[0]
        return [
            hit
            for hit in self.vector_store.query(embedding, top_k=self.top_k)
            if hit.relevance >= self.minimum_relevance
        ]

