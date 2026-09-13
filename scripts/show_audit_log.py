"""Verify and show recent integrity-protected audit events."""

from __future__ import annotations

import argparse
import json

from clinical_assistant.audit.audit_logger import JsonlAuditLogger
from clinical_assistant.config import load_yaml, project_root, resolve_project_path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Número de eventos recentes")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    root = project_root()
    config = load_yaml(root / "configs" / "safety.yaml")["audit"]
    logger = JsonlAuditLogger(
        resolve_project_path(config["path"], root=root),
        maximum_question_chars=int(config["maximum_question_chars"]),
        maximum_response_chars=int(config["maximum_response_chars"]),
        maximum_events_to_read=int(config["maximum_events_to_read"]),
    )
    result = {
        "integrity": logger.verify(),
        "recent_events": logger.read_events(limit=arguments.limit),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

