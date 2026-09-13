"""Build the stage 6 ChromaDB protocol index using CPU embeddings."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.rag.pipeline import build_rag_index


if __name__ == "__main__":
    result = build_rag_index(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))

