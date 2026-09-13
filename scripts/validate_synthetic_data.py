"""Validate stage 3 synthetic datasets independently of their generator."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.synthetic import validate_synthetic_hospital


if __name__ == "__main__":
    result = validate_synthetic_hospital(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
