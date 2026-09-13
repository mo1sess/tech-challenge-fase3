"""Structured local patient data for stage 7."""

from clinical_assistant.database.builder import build_patient_database
from clinical_assistant.database.patient_repository import PatientRepository

__all__ = ["PatientRepository", "build_patient_database"]

