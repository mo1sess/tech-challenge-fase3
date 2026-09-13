from __future__ import annotations

from pathlib import Path

import pytest

from clinical_assistant.database.builder import build_patient_database
from clinical_assistant.database.patient_repository import PatientRepository, validate_patient_id
from tests.patient_database_helpers import create_stage7_fixture


@pytest.fixture
def repository(tmp_path: Path) -> PatientRepository:
    root = create_stage7_fixture(tmp_path)
    build_patient_database(root)
    return PatientRepository(root / "data" / "database" / "techcare.db")


@pytest.mark.unit
def test_patient_id_accepts_only_pseudonymized_format() -> None:
    assert validate_patient_id(" pac001 ") == "PAC001"
    for unsafe in ("PAC001' OR 1=1 --", "1", "Maria", "PAC-001"):
        with pytest.raises(ValueError):
            validate_patient_id(unsafe)


@pytest.mark.integration
def test_repository_reads_patient_context(repository: PatientRepository) -> None:
    patient = repository.get_patient("PAC001")
    assert patient is not None
    assert patient["birth_year"] == 1980
    assert patient["synthetic"] == 1
    assert repository.get_patient_conditions("PAC001")[0]["description"] == "Asthma"
    assert repository.get_patient_medications("PAC001")[0]["source"] == "Synthea anonimizado"
    assert repository.get_patient_observations("PAC001")[0]["value"] == "72"
    assert repository.get_pending_exams("PAC001")[0]["status"] == "pending"


@pytest.mark.unit
def test_repository_enforces_result_limit(repository: PatientRepository) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        repository.get_patient_observations("PAC001", limit=101)
    with pytest.raises(TypeError, match="integer"):
        repository.get_patient_observations("PAC001", limit=True)


@pytest.mark.unit
def test_repository_does_not_expose_arbitrary_sql(repository: PatientRepository) -> None:
    assert not hasattr(repository, "execute")
    assert not hasattr(repository, "query")
