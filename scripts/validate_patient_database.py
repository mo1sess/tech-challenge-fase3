"""Validate the stage 7 SQLite patient database and manifest."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.database import validate_patient_database


if __name__ == "__main__":
    report = validate_patient_database(project_root())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)

