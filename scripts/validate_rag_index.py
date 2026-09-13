"""Validate the stage 6 RAG artifacts and persistent Chroma collection."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.rag import validate_rag_index


if __name__ == "__main__":
    result = validate_rag_index(project_root(), check_store=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)

