"""Named patient tools with no arbitrary SQL surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.database.patient_repository import PatientRepository
from clinical_assistant.services.patient_service import PatientService


@dataclass(frozen=True)
class PatientTools:
    service: PatientService

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        return self.service.get_patient(patient_id)

    def get_patient_conditions(self, patient_id: str) -> dict[str, Any]:
        return self.service.get_patient_conditions(patient_id)

    def get_patient_medications(self, patient_id: str) -> dict[str, Any]:
        return self.service.get_patient_medications(patient_id)

    def get_pending_exams(self, patient_id: str) -> dict[str, Any]:
        return self.service.get_pending_exams(patient_id)

    def get_patient_observations(self, patient_id: str) -> dict[str, Any]:
        return self.service.get_patient_observations(patient_id)

    def get_patient_summary(self, patient_id: str) -> dict[str, Any]:
        return self.service.get_patient_summary(patient_id)


def create_patient_tools(root: Path) -> PatientTools:
    config = load_yaml(root / "configs" / "database.yaml")
    repository = PatientRepository(
        resolve_project_path(config["database"]["path"], root=root),
        maximum_result_limit=int(config["database"]["maximum_result_limit"]),
    )
    return PatientTools(
        PatientService(
            repository,
            default_limit=int(config["database"]["default_result_limit"]),
        )
    )

