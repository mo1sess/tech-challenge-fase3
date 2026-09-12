"""Validate the datasets acquired in stage 1."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.datasets import validate_all


if __name__ == "__main__":
    report = validate_all(project_root())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)

