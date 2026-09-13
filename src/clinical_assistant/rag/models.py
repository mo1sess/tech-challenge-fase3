"""Typed records shared by the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RagDocument:
    document_id: str
    document_name: str
    version: str
    section: str
    source: str
    document_type: str
    text: str
    synthetic: bool
    notice: str

    def metadata(self) -> dict[str, str | bool]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "version": self.version,
            "section": self.section,
            "source": self.source,
            "document_type": self.document_type,
            "synthetic": self.synthetic,
            "notice": self.notice,
        }


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    relevance: float

    @property
    def citation(self) -> str:
        name = self.metadata["document_name"]
        version = self.metadata["version"]
        section = self.metadata["section"]
        source = self.metadata["source"]
        return f"{name} — versão {version}, seção {section}; fonte: {source}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["citation"] = self.citation
        return value

