from __future__ import annotations

import csv
from pathlib import Path

import pytest

from clinical_assistant.preprocessing.anonymizer import (
    anonymize_synthea_directory,
    anonymize_text,
    build_patient_id_map,
)


@pytest.mark.unit
def test_anonymize_text_masks_direct_identifiers() -> None:
    source = (
        "Nome: Maria da Silva\n"
        "CPF: 123.456.789-00; telefone: (11) 99876-5432; "
        "e-mail maria@example.com\nEndereco: Rua Um, 10"
    )
    result = anonymize_text(source)
    assert "Maria da Silva" not in result
    assert "123.456.789-00" not in result
    assert "99876-5432" not in result
    assert "maria@example.com" not in result
    assert "Rua Um" not in result
    assert "[NOME]" in result
    assert "[CPF]" in result
    assert "[TELEFONE]" in result
    assert "[EMAIL]" in result
    assert "[ENDERECO]" in result


@pytest.mark.unit
def test_anonymizer_preserves_technical_patient_id() -> None:
    assert anonymize_text("Paciente PAC001") == "Paciente PAC001"


@pytest.mark.unit
def test_patient_map_is_stable_and_sequential() -> None:
    assert build_patient_id_map(["uuid-b", "uuid-a", "uuid-b"]) == {
        "uuid-b": "PAC001",
        "uuid-a": "PAC002",
    }


@pytest.mark.integration
def test_synthea_anonymization_preserves_relations(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    patient_fields = [
        "Id", "BIRTHDATE", "DEATHDATE", "SSN", "DRIVERS", "PASSPORT",
        "PREFIX", "FIRST", "MIDDLE", "LAST", "SUFFIX", "MAIDEN", "MARITAL",
        "RACE", "ETHNICITY", "GENDER", "BIRTHPLACE", "ADDRESS", "CITY", "STATE",
        "COUNTY", "FIPS", "ZIP", "LAT", "LON", "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE", "INCOME",
    ]
    with (raw / "patients.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=patient_fields)
        writer.writeheader()
        writer.writerow(
            {
                "Id": "secret-uuid", "BIRTHDATE": "1990-05-03", "SSN": "999-00-0000",
                "FIRST": "Maria", "LAST": "Silva", "ADDRESS": "Rua Um", "CITY": "Boston",
                "STATE": "Massachusetts", "COUNTY": "Suffolk", "RACE": "white",
                "ETHNICITY": "nonhispanic", "GENDER": "F", "INCOME": "50000",
            }
        )
    with (raw / "conditions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["PATIENT", "DESCRIPTION"])
        writer.writeheader()
        writer.writerow({"PATIENT": "secret-uuid", "DESCRIPTION": "Asthma"})

    result = anonymize_synthea_directory(raw, processed)
    assert result["patients"] == 1
    patients_text = (processed / "patients.csv").read_text(encoding="utf-8")
    conditions_text = (processed / "conditions.csv").read_text(encoding="utf-8")
    assert "secret-uuid" not in patients_text + conditions_text
    assert "Maria" not in patients_text
    assert "999-00-0000" not in patients_text
    assert "PAC001" in patients_text
    assert "PAC001" in conditions_text
    assert "BIRTH_YEAR" in patients_text
    assert "BIRTHDATE" not in patients_text


@pytest.mark.unit
def test_synthea_rejects_unknown_patient_reference(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with (raw / "patients.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Id", "BIRTHDATE"])
        writer.writeheader()
        writer.writerow({"Id": "known", "BIRTHDATE": "2000-01-01"})
    with (raw / "conditions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["PATIENT"])
        writer.writeheader()
        writer.writerow({"PATIENT": "unknown"})
    with pytest.raises(ValueError, match="Unknown patient reference"):
        anonymize_synthea_directory(raw, tmp_path / "processed")

