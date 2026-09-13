"""Validate the local stage 9 guardrails, audit configuration and adversarial suite."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.safety import validate_safety_stage


if __name__ == "__main__":
    result = validate_safety_stage(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)

