"""Read-only, parameterized access to pseudonymized patient records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from clinical_assistant.database.database import database_connection


PATIENT_ID_PATTERN = re.compile(r"^PAC[0-9]{3,}$")


def validate_patient_id(patient_id: str) -> str:
    normalized = patient_id.strip().upper()
    if not PATIENT_ID_PATTERN.fullmatch(normalized):
        raise ValueError("patient_id must use the pseudonymized format PACnnn")
    return normalized


class PatientRepository:
    """Expose only fixed read queries; arbitrary SQL is intentionally absent."""

    def __init__(self, database_path: Path, *, maximum_result_limit: int = 100) -> None:
        if maximum_result_limit < 1:
            raise ValueError("maximum_result_limit must be positive")
        self.database_path = database_path
        self.maximum_result_limit = maximum_result_limit

    def _limit(self, limit: int) -> int:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        if limit < 1 or limit > self.maximum_result_limit:
            raise ValueError(
                f"limit must be between 1 and {self.maximum_result_limit}"
            )
        return limit

    @staticmethod
    def _rows(rows) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        patient_id = validate_patient_id(patient_id)
        with database_connection(self.database_path, read_only=True) as connection:
            row = connection.execute(
                """SELECT patient_id, birth_year, death_year, marital, race,
                          ethnicity, gender, state, county, source, synthetic
                   FROM patients WHERE patient_id = ?""",
                (patient_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_patient_conditions(
        self, patient_id: str, *, active_only: bool = False, limit: int = 20
    ) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        limit = self._limit(limit)
        active_clause = "AND (ended_at IS NULL OR ended_at = '')" if active_only else ""
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                f"""SELECT started_at, ended_at, code_system, code, description,
                           encounter_id, source, synthetic
                    FROM conditions
                    WHERE patient_id = ? {active_clause}
                    ORDER BY started_at DESC, id DESC LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        return self._rows(rows)

    def get_patient_medications(
        self, patient_id: str, *, active_only: bool = False, limit: int = 20
    ) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        limit = self._limit(limit)
        active_clause = "AND (ended_at IS NULL OR ended_at = '')" if active_only else ""
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                f"""SELECT started_at, ended_at, code, description, reason_code,
                           reason_description, encounter_id, source, synthetic
                    FROM medications
                    WHERE patient_id = ? {active_clause}
                    ORDER BY started_at DESC, id DESC LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        return self._rows(rows)

    def get_pending_exams(self, patient_id: str) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                """SELECT exam_request_id, exam_code, description, requested_at,
                          due_at, status, source, synthetic, notice
                   FROM pending_exams
                   WHERE patient_id = ? AND status = 'pending'
                   ORDER BY due_at, exam_request_id""",
                (patient_id,),
            ).fetchall()
        return self._rows(rows)

    def get_patient_observations(
        self, patient_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        limit = self._limit(limit)
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                """SELECT observed_at, category, code, description, value, units,
                          value_type, encounter_id, source, synthetic
                   FROM observations WHERE patient_id = ?
                   ORDER BY observed_at DESC, id DESC LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        return self._rows(rows)

    def get_patient_encounters(
        self, patient_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        limit = self._limit(limit)
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                """SELECT encounter_id, started_at, ended_at, encounter_class,
                          code, description, reason_code, reason_description,
                          source, synthetic
                   FROM encounters WHERE patient_id = ?
                   ORDER BY started_at DESC, encounter_id LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        return self._rows(rows)

    def get_patient_procedures(
        self, patient_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        limit = self._limit(limit)
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                """SELECT started_at, ended_at, code_system, code, description,
                          reason_code, reason_description, encounter_id, source,
                          synthetic
                   FROM procedures WHERE patient_id = ?
                   ORDER BY started_at DESC, id DESC LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        return self._rows(rows)

    def get_patient_allergies(
        self, patient_id: str, *, active_only: bool = False, limit: int = 20
    ) -> list[dict[str, Any]]:
        patient_id = validate_patient_id(patient_id)
        limit = self._limit(limit)
        active_clause = "AND (ended_at IS NULL OR ended_at = '')" if active_only else ""
        with database_connection(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                f"""SELECT started_at, ended_at, code_system, code, description,
                           allergy_type, category, reaction_description, severity,
                           encounter_id, source, synthetic
                    FROM allergies
                    WHERE patient_id = ? {active_clause}
                    ORDER BY started_at DESC, id DESC LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        return self._rows(rows)

