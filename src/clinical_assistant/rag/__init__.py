"""Local protocol retrieval for stage 6."""

from clinical_assistant.rag.chunker import chunk_documents
from clinical_assistant.rag.loader import load_synthetic_protocols
from clinical_assistant.rag.models import RagChunk, RagDocument, RetrievalHit
from clinical_assistant.rag.retriever import ProtocolRetriever

__all__ = [
    "ProtocolRetriever",
    "RagChunk",
    "RagDocument",
    "RetrievalHit",
    "chunk_documents",
    "load_synthetic_protocols",
]

