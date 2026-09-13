"""CPU embedding providers used by production and isolated tests."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingProvider:
    """Lazy, CPU-only multilingual embeddings for the local RAG index."""

    def __init__(self, *, model_id: str, revision: str, device: str = "cpu") -> None:
        if device.lower() != "cpu":
            raise ValueError("Stage 6 embeddings are restricted to CPU")
        self.model_id = model_id
        self.revision = revision
        self.device = "cpu"
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install requirements/rag-local.txt"
                ) from exc
            self._model = SentenceTransformer(
                self.model_id,
                revision=self.revision,
                device=self.device,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


class DeterministicHashEmbeddingProvider:
    """Small offline embedding used only by tests, never by the official index."""

    model_id = "test/deterministic-hash-embedding"
    revision = "1"

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                position = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[position] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            result.append([value / norm for value in vector])
        return result

