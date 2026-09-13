"""Run one controlled patient tool against the local SQLite database."""

from __future__ import annotations

import argparse
import json

from clinical_assistant.config import project_root
from clinical_assistant.tools.patient_tools import create_patient_tools


OPERATIONS = {
    "patient": "get_patient",
    "summary": "get_patient_summary",
    "conditions": "get_patient_conditions",
    "medications": "get_patient_medications",
    "observations": "get_patient_observations",
    "pending-exams": "get_pending_exams",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_id", help="Identificador pseudonimizado, por exemplo PAC004")
    parser.add_argument("operation", choices=sorted(OPERATIONS), nargs="?", default="summary")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    tools = create_patient_tools(project_root())
    result = getattr(tools, OPERATIONS[arguments.operation])(arguments.patient_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))

