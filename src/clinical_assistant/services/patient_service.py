"""Safety-oriented patient context service."""

from __future__ import annotations

from typing import Any, Callable

from clinical_assistant.database.patient_repository import PatientRepository


DISCLAIMER = (
    "Dados sintéticos para demonstração acadêmica. Não usar para diagnóstico, "
    "prescrição ou decisão clínica autônoma. Validação médica necessária."
)


class PatientService:
    def __init__(self, repository: PatientRepository, *, default_limit: int = 20) -> None:
        self.repository = repository
        self.default_limit = default_limit

    @staticmethod
    def _result(
        patient_id: str,
        operation: str,
        data: Any,
        *,
        found: bool = True,
    ) -> dict[str, Any]:
        return {
            "ok": found,
            "patient_id": patient_id.strip().upper(),
            "operation": operation,
            "data": data,
            "source": "Prontuário sintético Synthea anonimizado / Hospital TechCare fictício",
            "synthetic": True,
            "disclaimer": DISCLAIMER,
        }

    def _patient_required(
        self,
        patient_id: str,
        operation: str,
        query: Callable[[], Any],
    ) -> dict[str, Any]:
        patient = self.repository.get_patient(patient_id)
        if patient is None:
            return self._result(
                patient_id,
                operation,
                {"error": "patient_not_found"},
                found=False,
            )
        return self._result(patient_id, operation, query())

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        patient = self.repository.get_patient(patient_id)
        return self._result(
            patient_id,
            "get_patient",
            patient if patient is not None else {"error": "patient_not_found"},
            found=patient is not None,
        )

    def get_patient_conditions(self, patient_id: str) -> dict[str, Any]:
        return self._patient_required(
            patient_id,
            "get_patient_conditions",
            lambda: self.repository.get_patient_conditions(
                patient_id, active_only=False, limit=self.default_limit
            ),
        )

    def get_patient_medications(self, patient_id: str) -> dict[str, Any]:
        return self._patient_required(
            patient_id,
            "get_patient_medications",
            lambda: self.repository.get_patient_medications(
                patient_id, active_only=False, limit=self.default_limit
            ),
        )

    def get_pending_exams(self, patient_id: str) -> dict[str, Any]:
        return self._patient_required(
            patient_id,
            "get_pending_exams",
            lambda: self.repository.get_pending_exams(patient_id),
        )

    def get_patient_observations(self, patient_id: str) -> dict[str, Any]:
        return self._patient_required(
            patient_id,
            "get_patient_observations",
            lambda: self.repository.get_patient_observations(
                patient_id, limit=self.default_limit
            ),
        )

    def get_patient_summary(self, patient_id: str) -> dict[str, Any]:
        patient = self.repository.get_patient(patient_id)
        if patient is None:
            return self._result(
                patient_id,
                "get_patient_summary",
                {"error": "patient_not_found"},
                found=False,
            )
        data = {
            "patient": patient,
            "conditions": self.repository.get_patient_conditions(
                patient_id, active_only=True, limit=self.default_limit
            ),
            "medications": self.repository.get_patient_medications(
                patient_id, active_only=True, limit=self.default_limit
            ),
            "observations": self.repository.get_patient_observations(
                patient_id, limit=min(10, self.default_limit)
            ),
            "pending_exams": self.repository.get_pending_exams(patient_id),
        }
        return self._result(patient_id, "get_patient_summary", data)

