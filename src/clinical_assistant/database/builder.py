"""Build a deterministic SQLite database from anonymized Synthea CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.database.database import connect_database
from clinical_assistant.database.schema import SCHEMA_VERSION, create_schema


SYNTHEA_SOURCE = "Synthea anonimizado"
ACADEMIC_NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hash(table: str, row: dict[str, str], source_row_number: int) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(
        f"{table}:{source_row_number}:{payload}".encode("utf-8")
    ).hexdigest()


def _none(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def _integer(value: str | None) -> int | None:
    cleaned = _none(value)
    return int(cleaned) if cleaned is not None else None


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required anonymized table not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _insert_many(
    connection: sqlite3.Connection,
    sql: str,
    rows: Iterable[tuple[Any, ...]],
    *,
    batch_size: int = 2000,
) -> int:
    batch: list[tuple[Any, ...]] = []
    count = 0
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            connection.executemany(sql, batch)
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
        count += len(batch)
    return count


def _patient_rows(path: Path) -> Iterable[tuple[Any, ...]]:
    for row in _read_csv(path):
        yield (
            row["Id"],
            _integer(row.get("BIRTH_YEAR")),
            _integer(row.get("DEATH_YEAR")),
            _none(row.get("MARITAL")),
            _none(row.get("RACE")),
            _none(row.get("ETHNICITY")),
            _none(row.get("GENDER")),
            _none(row.get("STATE")),
            _none(row.get("COUNTY")),
            SYNTHEA_SOURCE,
            1,
        )


def _related_rows(
    path: Path,
    table: str,
    converter: Callable[[dict[str, str]], tuple[Any, ...]],
) -> Iterable[tuple[Any, ...]]:
    for source_row_number, row in enumerate(_read_csv(path), start=2):
        yield (
            _row_hash(table, row, source_row_number),
            *converter(row),
            SYNTHEA_SOURCE,
            1,
        )


def _load_pending_exams(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pending-exam seed not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("synthetic") is not True or value.get("notice") != ACADEMIC_NOTICE:
            raise ValueError(f"Pending exam line {line_number} lacks the synthetic notice")
        if value.get("status") != "pending":
            raise ValueError(f"Pending exam line {line_number} has a non-pending status")
        records.append(value)
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_patient_database(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "database.yaml"
    config = load_yaml(config_path)
    source = resolve_project_path(config["source"]["synthea_directory"], root=root)
    pending_path = resolve_project_path(config["source"]["pending_exams_file"], root=root)
    database_path = resolve_project_path(config["database"]["path"], root=root)
    manifest_path = resolve_project_path(config["database"]["manifest"], root=root)

    if database_path.exists():
        if not database_path.is_file():
            raise ValueError(f"Database target is not a file: {database_path}")
        database_path.unlink()

    connection = connect_database(database_path)
    counts: dict[str, int] = {}
    try:
        connection.execute(f"PRAGMA journal_mode = {config['database']['journal_mode']}")
        create_schema(connection)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("stage", "7"),
                ("source", SYNTHEA_SOURCE),
                ("synthetic", "true"),
            ],
        )
        counts["patients"] = _insert_many(
            connection,
            """INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            _patient_rows(source / "patients.csv"),
        )
        counts["encounters"] = _insert_many(
            connection,
            """INSERT INTO encounters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                (
                    row["Id"], row["PATIENT"], _none(row.get("START")),
                    _none(row.get("STOP")), _none(row.get("ENCOUNTERCLASS")),
                    _none(row.get("CODE")), _none(row.get("DESCRIPTION")),
                    _none(row.get("REASONCODE")), _none(row.get("REASONDESCRIPTION")),
                    SYNTHEA_SOURCE, 1,
                )
                for row in _read_csv(source / "encounters.csv")
            ),
        )

        related: list[tuple[str, str, str, Callable[[dict[str, str]], tuple[Any, ...]]]] = [
            (
                "conditions",
                "INSERT INTO conditions(source_row_hash, patient_id, encounter_id, started_at, ended_at, code_system, code, description, source, synthetic) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                "conditions.csv",
                lambda r: (r["PATIENT"], _none(r.get("ENCOUNTER")), _none(r.get("START")), _none(r.get("STOP")), _none(r.get("SYSTEM")), _none(r.get("CODE")), _none(r.get("DESCRIPTION"))),
            ),
            (
                "observations",
                "INSERT INTO observations(source_row_hash, patient_id, encounter_id, observed_at, category, code, description, value, units, value_type, source, synthetic) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                "observations.csv",
                lambda r: (r["PATIENT"], _none(r.get("ENCOUNTER")), _none(r.get("DATE")), _none(r.get("CATEGORY")), _none(r.get("CODE")), _none(r.get("DESCRIPTION")), _none(r.get("VALUE")), _none(r.get("UNITS")), _none(r.get("TYPE"))),
            ),
            (
                "medications",
                "INSERT INTO medications(source_row_hash, patient_id, encounter_id, started_at, ended_at, code, description, reason_code, reason_description, source, synthetic) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                "medications.csv",
                lambda r: (r["PATIENT"], _none(r.get("ENCOUNTER")), _none(r.get("START")), _none(r.get("STOP")), _none(r.get("CODE")), _none(r.get("DESCRIPTION")), _none(r.get("REASONCODE")), _none(r.get("REASONDESCRIPTION"))),
            ),
            (
                "procedures",
                "INSERT INTO procedures(source_row_hash, patient_id, encounter_id, started_at, ended_at, code_system, code, description, reason_code, reason_description, source, synthetic) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                "procedures.csv",
                lambda r: (r["PATIENT"], _none(r.get("ENCOUNTER")), _none(r.get("START")), _none(r.get("STOP")), _none(r.get("SYSTEM")), _none(r.get("CODE")), _none(r.get("DESCRIPTION")), _none(r.get("REASONCODE")), _none(r.get("REASONDESCRIPTION"))),
            ),
            (
                "allergies",
                "INSERT INTO allergies(source_row_hash, patient_id, encounter_id, started_at, ended_at, code_system, code, description, allergy_type, category, reaction_description, severity, source, synthetic) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                "allergies.csv",
                lambda r: (r["PATIENT"], _none(r.get("ENCOUNTER")), _none(r.get("START")), _none(r.get("STOP")), _none(r.get("SYSTEM")), _none(r.get("CODE")), _none(r.get("DESCRIPTION")), _none(r.get("TYPE")), _none(r.get("CATEGORY")), _none(r.get("DESCRIPTION1")), _none(r.get("SEVERITY1"))),
            ),
        ]
        for table, sql, filename, converter in related:
            counts[table] = _insert_many(
                connection,
                sql,
                _related_rows(source / filename, table, converter),
            )

        pending_records = _load_pending_exams(pending_path)
        counts["pending_exams"] = _insert_many(
            connection,
            """INSERT INTO pending_exams VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                (
                    item["exam_request_id"], item["patient_id"], item["exam_code"],
                    item["description"], item["requested_at"], item.get("due_at"),
                    item["status"], item["source"], 1, item["notice"],
                )
                for item in pending_records
            ),
        )
        expected = {key: int(value) for key, value in config["expected_counts"].items()}
        if counts != expected:
            raise ValueError(f"Imported counts do not match configuration: {counts} != {expected}")
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise ValueError(f"Foreign-key validation failed: {foreign_key_errors[:3]}")
        connection.commit()
        connection.execute("VACUUM")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
    except Exception:
        connection.rollback()
        connection.close()
        if database_path.exists():
            database_path.unlink()
        raise
    finally:
        if connection:
            connection.close()

    source_files = {
        f"{table}.csv": _sha256(source / f"{table}.csv")
        for table in ("patients", "encounters", "conditions", "observations", "medications", "procedures", "allergies")
    }
    source_files[pending_path.name] = _sha256(pending_path)
    manifest = {
        "schema_version": 1,
        "stage": 7,
        "complete": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "database": {
            "path": config["database"]["path"],
            "sha256": _sha256(database_path),
            "bytes": database_path.stat().st_size,
            "sqlite_schema_version": SCHEMA_VERSION,
        },
        "sources": {
            "synthea_directory": config["source"]["synthea_directory"],
            "pending_exams_file": config["source"]["pending_exams_file"],
            "files_sha256": source_files,
            "synthetic": True,
            "anonymized": True,
        },
        "table_counts": counts,
        "safety": {
            "repository_read_only": True,
            "arbitrary_sql_allowed": False,
            "patient_identifier": "pseudonymized PACnnn",
            "direct_identifiers_imported": False,
        },
        "limitations": [
            "Todos os prontuários são sintéticos e não representam pessoas reais.",
            "Pendências são registros acadêmicos explícitos; não são inferidas pela ausência de resultado.",
            "As consultas apoiam demonstração técnica e não autorizam decisão clínica.",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest
