"""Build the stage 7 SQLite patient database from anonymized synthetic data."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.database.builder import build_patient_database


if __name__ == "__main__":
    print(json.dumps(build_patient_database(project_root()), ensure_ascii=False, indent=2))

