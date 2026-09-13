from __future__ import annotations

import math

import pytest

from clinical_assistant.rag.embeddings import (
    DeterministicHashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


@pytest.mark.unit
def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    provider = DeterministicHashEmbeddingProvider(dimensions=32)
    first = provider.encode(["exames pendentes"])[0]
    second = provider.encode(["exames pendentes"])[0]
    assert first == second
    assert len(first) == 32
    assert math.isclose(sum(value * value for value in first), 1.0)


@pytest.mark.unit
def test_official_embedding_provider_refuses_gpu() -> None:
    with pytest.raises(ValueError, match="restricted to CPU"):
        SentenceTransformerEmbeddingProvider(
            model_id="model",
            revision="revision",
            device="cuda",
        )

