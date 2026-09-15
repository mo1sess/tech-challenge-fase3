"""Validate the local preflight for the official remote-agent integration."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.full_agent import validate_full_agent_preflight


if __name__ == "__main__":
    result = validate_full_agent_preflight(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
