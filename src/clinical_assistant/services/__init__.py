"""Application services kept independent from orchestration frameworks."""

from clinical_assistant.services.patient_service import PatientService

__all__ = ["PatientService"]

