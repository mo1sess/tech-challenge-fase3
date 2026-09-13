from __future__ import annotations

from pathlib import Path

import pytest

from clinical_assistant.database.builder import build_patient_database
from clinical_assistant.database.patient_repository import PatientRepository
from clinical_assistant.services.patient_service import PatientService
from tests.patient_database_helpers import NOTICE, create_stage7_fixture


@pytest.fixture
def service(tmp_path: Path) -> PatientService:
    root = create_stage7_fixture(tmp_path)
    build_patient_database(root)
    return PatientService(PatientRepository(root / "data" / "database" / "techcare.db"))


@pytest.mark.integration
def test_service_summary_is_traceable_and_safe(service: PatientService) -> None:
    result = service.get_patient_summary("PAC001")
    assert result["ok"] is True
    assert result["synthetic"] is True
    assert "Validação médica necessária" in result["disclaimer"]
    assert result["data"]["pending_exams"][0]["notice"] == NOTICE


@pytest.mark.unit
def test_service_reports_unknown_patient_without_fabricating_data(service: PatientService) -> None:
    result = service.get_patient_conditions("PAC999")
    assert result["ok"] is False
    assert result["data"] == {"error": "patient_not_found"}


@pytest.mark.unit
def test_service_rejects_unsafe_patient_identifier(service: PatientService) -> None:
    with pytest.raises(ValueError):
        service.get_patient("PAC001; DROP TABLE patients")
