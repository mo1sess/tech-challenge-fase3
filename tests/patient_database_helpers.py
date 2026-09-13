from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


NOTICE = "DOCUMENTO SINTÉTICO PARA FINS ACADÊMICOS"


TABLES: dict[str, tuple[list[str], list[str]]] = {
    "patients": (
        ["Id", "BIRTH_YEAR", "DEATH_YEAR", "MARITAL", "RACE", "ETHNICITY", "GENDER", "STATE", "COUNTY"],
        ["PAC001", "1980", "", "M", "white", "nonhispanic", "F", "SP", "São Paulo"],
    ),
    "encounters": (
        ["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS", "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
        ["ENC001", "2026-01-02", "2026-01-02", "PAC001", "ambulatory", "185349003", "Consulta", "", ""],
    ),
    "conditions": (
        ["START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION"],
        ["2025-01-01", "", "PAC001", "ENC001", "SNOMED-CT", "195967001", "Asthma"],
    ),
    "observations": (
        ["DATE", "PATIENT", "ENCOUNTER", "CATEGORY", "CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE"],
        ["2026-01-02", "PAC001", "ENC001", "vital-signs", "8867-4", "Heart rate", "72", "beats/min", "numeric"],
    ),
    "medications": (
        ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
        ["2026-01-02", "", "PAC001", "ENC001", "123", "Medicamento sintético", "195967001", "Asthma"],
    ),
    "procedures": (
        ["START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
        ["2026-01-02", "2026-01-02", "PAC001", "ENC001", "SNOMED-CT", "456", "Procedimento sintético", "", ""],
    ),
    "allergies": (
        ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "SYSTEM", "DESCRIPTION", "TYPE", "CATEGORY", "DESCRIPTION1", "SEVERITY1"],
        ["2024-01-01", "", "PAC001", "ENC001", "789", "SNOMED-CT", "Substância sintética", "allergy", "environment", "Rinite", "mild"],
    ),
}


def create_stage7_fixture(root: Path) -> Path:
    source = root / "data" / "processed" / "synthea_anonymized"
    source.mkdir(parents=True)
    for table, (header, row) in TABLES.items():
        with (source / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerow(row)

    pending = root / "data" / "synthetic" / "hospital" / "pending_exams.jsonl"
    pending.parent.mkdir(parents=True)
    pending.write_text(
        json.dumps(
            {
                "exam_request_id": "DEMO-001",
                "patient_id": "PAC001",
                "exam_code": "DEMO",
                "description": "Exame sintético pendente",
                "requested_at": "2026-01-02",
                "due_at": "2026-02-02",
                "status": "pending",
                "source": "Hospital TechCare fictício",
                "synthetic": True,
                "notice": NOTICE,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "stage": 7,
        "source": {
            "synthea_directory": "data/processed/synthea_anonymized",
            "pending_exams_file": "data/synthetic/hospital/pending_exams.jsonl",
            "preprocessing_manifest": "data/processed/preprocessing_manifest.json",
        },
        "database": {
            "path": "data/database/techcare.db",
            "manifest": "outputs/database/database_manifest.json",
            "journal_mode": "DELETE",
            "patient_id_pattern": "^PAC[0-9]{3,}$",
            "default_result_limit": 20,
            "maximum_result_limit": 100,
        },
        "expected_counts": {table: 1 for table in (*TABLES, "pending_exams")},
        "required_tables": [*TABLES, "pending_exams", "metadata"],
        "safety": {
            "repository_read_only": True,
            "allow_arbitrary_sql": False,
            "direct_identifiers_forbidden": ["ssn", "first", "last", "address"],
        },
    }
    configs = root / "configs"
    configs.mkdir()
    (configs / "database.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return root
