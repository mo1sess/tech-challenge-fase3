"""Append-only JSONL audit logger with a SHA-256 integrity chain."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64
REQUIRED_FIELDS = {
    "execution_id",
    "timestamp",
    "patient_id",
    "question",
    "tools_called",
    "retrieved_documents",
    "sources",
    "model",
    "response",
    "safety_result",
    "human_validation_required",
    "human_validation_result",
}


class AuditIntegrityError(ValueError):
    pass


class JsonlAuditLogger:
    _registry_lock = threading.Lock()
    _path_locks: dict[str, threading.RLock] = {}

    def __init__(
        self,
        path: Path,
        *,
        maximum_question_chars: int = 1000,
        maximum_response_chars: int = 12000,
        maximum_events_to_read: int = 1000,
    ) -> None:
        if min(maximum_question_chars, maximum_response_chars, maximum_events_to_read) < 1:
            raise ValueError("audit limits must be positive")
        self.path = path
        self.maximum_question_chars = maximum_question_chars
        self.maximum_response_chars = maximum_response_chars
        self.maximum_events_to_read = maximum_events_to_read
        with self._registry_lock:
            self._lock = self._path_locks.setdefault(str(path.resolve()), threading.RLock())

    @staticmethod
    def _canonical(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _hash(cls, value: dict[str, Any]) -> str:
        return hashlib.sha256(cls._canonical(value).encode("utf-8")).hexdigest()

    def _raw_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditIntegrityError(f"invalid JSON at audit line {line_number}") from exc
                if not isinstance(event, dict):
                    raise AuditIntegrityError(f"audit line {line_number} is not an object")
                events.append(event)
        return events

    def verify(self) -> dict[str, Any]:
        with self._lock:
            events = self._raw_events()
            previous = GENESIS_HASH
            execution_ids: set[str] = set()
            for index, event in enumerate(events, 1):
                missing = REQUIRED_FIELDS - set(event)
                if missing:
                    raise AuditIntegrityError(
                        f"audit line {index} missing fields: {sorted(missing)}"
                    )
                if event.get("previous_hash") != previous:
                    raise AuditIntegrityError(f"audit chain broken at line {index}")
                recorded_hash = event.get("event_hash")
                payload = {key: value for key, value in event.items() if key != "event_hash"}
                if recorded_hash != self._hash(payload):
                    raise AuditIntegrityError(f"audit hash mismatch at line {index}")
                execution_id = str(event["execution_id"])
                if execution_id in execution_ids:
                    raise AuditIntegrityError(f"duplicate execution_id at line {index}")
                execution_ids.add(execution_id)
                previous = str(recorded_hash)
            return {
                "ok": True,
                "events": len(events),
                "last_hash": previous,
                "path": str(self.path),
            }

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        missing = REQUIRED_FIELDS - set(event)
        if missing:
            raise ValueError(f"audit event missing fields: {sorted(missing)}")
        with self._lock:
            verification = self.verify()
            if any(
                str(saved["execution_id"]) == str(event["execution_id"])
                for saved in self._raw_events()
            ):
                raise ValueError(f"duplicate execution_id: {event['execution_id']}")
            payload = {
                "schema_version": 1,
                "event_type": str(event.get("event_type", "workflow_completed")),
                "execution_id": str(event["execution_id"]),
                "timestamp": str(event.get("timestamp") or datetime.now(UTC).isoformat()),
                "patient_id": str(event["patient_id"]),
                "question": str(event["question"])[: self.maximum_question_chars],
                "tools_called": list(event["tools_called"]),
                "retrieved_documents": list(event["retrieved_documents"]),
                "sources": list(event["sources"]),
                "model": str(event["model"]),
                "response": str(event["response"])[: self.maximum_response_chars],
                "safety_result": dict(event["safety_result"]),
                "human_validation_required": bool(event["human_validation_required"]),
                "human_validation_result": event["human_validation_result"],
                "previous_hash": verification["last_hash"],
            }
            payload["event_hash"] = self._hash(payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(self._canonical(payload) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return {
                "ok": True,
                "path": str(self.path),
                "execution_id": payload["execution_id"],
                "event_hash": payload["event_hash"],
                "event_number": verification["events"] + 1,
            }

    def read_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        if limit < 1 or limit > self.maximum_events_to_read:
            raise ValueError(f"limit must be between 1 and {self.maximum_events_to_read}")
        with self._lock:
            self.verify()
            return self._raw_events()[-limit:]
