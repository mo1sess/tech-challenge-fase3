"""Validate the held-out stage 4 evaluation suite."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.evaluation.dataset import validate_evaluation_dataset


if __name__ == "__main__":
    result = validate_evaluation_dataset(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
