"""Validate stage 2 outputs independently of their generation pipeline."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.preprocessing import validate_processed


if __name__ == "__main__":
    result = validate_processed(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)

