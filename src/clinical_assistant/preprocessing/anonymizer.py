"""Explicit PII masking and relational pseudonymization for synthetic records."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable

from clinical_assistant.preprocessing.cleaner import normalize_text


EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
CPF_PATTERN = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+?55\s*)?\(?\d{2}\)?[\s.-]+(?:9\d{4}|\d{4})[\s.-]?\d{4}|(?:\+?55)?\d{2}9\d{8})(?!\d)"
)
NAME_LABEL_PATTERN = re.compile(
    r"\b(nome|name)\s*:\s*([^;\n]+)", re.IGNORECASE
)
ADDRESS_LABEL_PATTERN = re.compile(
    r"\b(endere[cç]o|address)\s*:\s*([^;\n]+)", re.IGNORECASE
)

PATIENT_REFERENCE_COLUMNS = {"PATIENT", "PATIENTID"}
SENSITIVE_REFERENCE_COLUMNS = {"MEMBERID", "PATIENTINSURANCEID"}
SAFE_PATIENT_COLUMNS = (
    "Id",
    "BIRTH_YEAR",
    "DEATH_YEAR",
    "MARITAL",
    "RACE",
    "ETHNICITY",
    "GENDER",
    "STATE",
    "COUNTY",
    "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE",
    "INCOME",
)


def anonymize_text(value: object) -> str:
    """Replace common direct identifiers while retaining surrounding meaning."""

    text = "" if value is None else str(value)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = CPF_PATTERN.sub("[CPF]", text)
    text = PHONE_PATTERN.sub("[TELEFONE]", text)
    text = NAME_LABEL_PATTERN.sub(lambda match: f"{match.group(1)}: [NOME]", text)
    text = ADDRESS_LABEL_PATTERN.sub(
        lambda match: f"{match.group(1)}: [ENDERECO]", text
    )
    return normalize_text(text)


def build_patient_id_map(
    identifiers: Iterable[str], *, prefix: str = "PAC"
) -> dict[str, str]:
    """Create stable sequential pseudonyms from source order without persisting PII."""

    unique = list(dict.fromkeys(value for value in identifiers if value))
    width = max(3, len(str(len(unique))))
    return {value: f"{prefix}{index:0{width}d}" for index, value in enumerate(unique, 1)}


def pseudonymize_reference(value: str, *, namespace: str) -> str:
    """Create a deterministic non-reversible reference for secondary identifiers."""

    if not value:
        return ""
    digest = hashlib.sha256(f"techcare-stage2:{namespace}:{value}".encode()).hexdigest()[:12]
    return f"{namespace}-{digest}"


def _year(value: str) -> str:
    return value[:4] if len(value) >= 4 and value[:4].isdigit() else ""


def _write_patients(
    source: Path, destination: Path, patient_map: dict[str, str]
) -> int:
    count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        with destination.open("w", encoding="utf-8", newline="") as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=SAFE_PATIENT_COLUMNS)
            writer.writeheader()
            for row in reader:
                writer.writerow(
                    {
                        "Id": patient_map[row["Id"]],
                        "BIRTH_YEAR": _year(row.get("BIRTHDATE", "")),
                        "DEATH_YEAR": _year(row.get("DEATHDATE", "")),
                        "MARITAL": row.get("MARITAL", ""),
                        "RACE": row.get("RACE", ""),
                        "ETHNICITY": row.get("ETHNICITY", ""),
                        "GENDER": row.get("GENDER", ""),
                        "STATE": row.get("STATE", ""),
                        "COUNTY": row.get("COUNTY", ""),
                        "HEALTHCARE_EXPENSES": row.get("HEALTHCARE_EXPENSES", ""),
                        "HEALTHCARE_COVERAGE": row.get("HEALTHCARE_COVERAGE", ""),
                        "INCOME": row.get("INCOME", ""),
                    }
                )
                count += 1
    return count


def _write_related_table(
    source: Path, destination: Path, patient_map: dict[str, str]
) -> tuple[int, set[str]]:
    count = 0
    referenced: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        fieldnames = reader.fieldnames or []
        with destination.open("w", encoding="utf-8", newline="") as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                cleaned: dict[str, str] = {}
                for column, raw_value in row.items():
                    value = raw_value or ""
                    if column.upper() in PATIENT_REFERENCE_COLUMNS:
                        if value and value not in patient_map:
                            raise ValueError(
                                f"Unknown patient reference in {source.name}: {value}"
                            )
                        if value:
                            value = patient_map[value]
                            referenced.add(value)
                    elif column.upper() in SENSITIVE_REFERENCE_COLUMNS:
                        value = pseudonymize_reference(value, namespace=column.upper())
                    elif column.upper() == "OWNER_NAME" and value:
                        value = "[OWNER]"
                    cleaned[column] = anonymize_text(value)
                writer.writerow(cleaned)
                count += 1
    return count, referenced


def anonymize_synthea_directory(
    source: Path, destination: Path, *, prefix: str = "PAC"
) -> dict[str, object]:
    """Anonymize every Synthea CSV and preserve patient foreign-key integrity."""

    patients_source = source / "patients.csv"
    if not patients_source.exists():
        raise FileNotFoundError(f"Missing Synthea patients.csv in {source}")
    destination.mkdir(parents=True, exist_ok=True)

    with patients_source.open("r", encoding="utf-8-sig", newline="") as stream:
        identifiers = [row["Id"] for row in csv.DictReader(stream)]
    patient_map = build_patient_id_map(identifiers, prefix=prefix)

    table_counts: dict[str, int] = {}
    all_references: set[str] = set()
    for csv_path in sorted(source.glob("*.csv")):
        target = destination / csv_path.name
        if csv_path.name.lower() == "patients.csv":
            table_counts[csv_path.stem] = _write_patients(csv_path, target, patient_map)
        else:
            count, references = _write_related_table(csv_path, target, patient_map)
            table_counts[csv_path.stem] = count
            all_references.update(references)

    pseudonyms = set(patient_map.values())
    if not all_references.issubset(pseudonyms):
        raise ValueError("Anonymized foreign keys are not a subset of patient IDs")
    return {
        "patients": len(patient_map),
        "tables": len(table_counts),
        "table_counts": table_counts,
        "patient_prefix": prefix,
        "direct_identifiers_removed": [
            "SSN",
            "DRIVERS",
            "PASSPORT",
            "PREFIX",
            "FIRST",
            "MIDDLE",
            "LAST",
            "SUFFIX",
            "MAIDEN",
            "BIRTHPLACE",
            "ADDRESS",
            "CITY",
            "FIPS",
            "ZIP",
            "LAT",
            "LON",
        ],
    }
