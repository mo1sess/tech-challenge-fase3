"""Independent validation of the stage 7 SQLite artifact."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.database.database import database_connection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_patient_database(root: Path) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "database.yaml")
    database_path = resolve_project_path(config["database"]["path"], root=root)
    manifest_path = resolve_project_path(config["database"]["manifest"], root=root)
    errors: list[str] = []
    counts: dict[str, int] = {}

    if not database_path.is_file():
        return {"ok": False, "stage": 7, "errors": [f"missing database: {database_path}"]}
    if not manifest_path.is_file():
        return {"ok": False, "stage": 7, "errors": [f"missing manifest: {manifest_path}"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "stage": 7, "errors": ["invalid manifest JSON"]}

    if manifest.get("stage") != 7 or manifest.get("complete") is not True:
        errors.append("manifest is not a complete stage 7 run")
    if manifest.get("database", {}).get("sha256") != _sha256(database_path):
        errors.append("database SHA-256 does not match manifest")

    required_tables = set(config["required_tables"])
    expected_counts = {key: int(value) for key, value in config["expected_counts"].items()}
    with database_connection(database_path, read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = sorted(required_tables - tables)
        if missing:
            errors.append(f"missing tables: {missing}")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            errors.append(f"integrity check failed: {integrity}")
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            errors.append(f"foreign-key errors: {len(foreign_key_errors)}")

        for table in expected_counts:
            if table in tables:
                counts[table] = int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
        if counts != expected_counts:
            errors.append(f"table counts differ: {counts} != {expected_counts}")
        if manifest.get("table_counts") != counts:
            errors.append("table counts do not match manifest")

        if "patients" in tables:
            columns = {
                row[1].casefold()
                for row in connection.execute("PRAGMA table_info(patients)").fetchall()
            }
            forbidden = set(config["safety"]["direct_identifiers_forbidden"])
            leaked_columns = sorted(columns & forbidden)
            if leaked_columns:
                errors.append(f"forbidden patient columns: {leaked_columns}")
            patient_ids = [row[0] for row in connection.execute("SELECT patient_id FROM patients")]
            pattern = re.compile(config["database"]["patient_id_pattern"])
            invalid_ids = [value for value in patient_ids if not pattern.fullmatch(value)]
            if invalid_ids:
                errors.append(f"invalid pseudonymized patient IDs: {invalid_ids[:3]}")

        if "pending_exams" in tables:
            unsafe_pending = connection.execute(
                """SELECT COUNT(*) FROM pending_exams
                   WHERE synthetic != 1 OR status != 'pending' OR notice != ?""",
                ("DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS",),
            ).fetchone()[0]
            if unsafe_pending:
                errors.append(f"invalid pending-exam records: {unsafe_pending}")

    return {
        "ok": not errors,
        "stage": 7,
        "database": str(database_path),
        "table_counts": counts,
        "integrity_check": "ok" if not any("integrity" in error for error in errors) else "failed",
        "foreign_key_errors": 0 if not any("foreign-key" in error for error in errors) else None,
        "direct_identifiers_imported": False if not any("forbidden" in error for error in errors) else None,
        "errors": errors,
    }

