"""SQLite schema for the minimal TechCare patient database."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    birth_year INTEGER,
    death_year INTEGER,
    marital TEXT,
    race TEXT,
    ethnicity TEXT,
    gender TEXT,
    state TEXT,
    county TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE encounters (
    encounter_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    started_at TEXT,
    ended_at TEXT,
    encounter_class TEXT,
    code TEXT,
    description TEXT,
    reason_code TEXT,
    reason_description TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row_hash TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    encounter_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    code_system TEXT,
    code TEXT,
    description TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row_hash TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    encounter_id TEXT,
    observed_at TEXT,
    category TEXT,
    code TEXT,
    description TEXT,
    value TEXT,
    units TEXT,
    value_type TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row_hash TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    encounter_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    code TEXT,
    description TEXT,
    reason_code TEXT,
    reason_description TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row_hash TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    encounter_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    code_system TEXT,
    code TEXT,
    description TEXT,
    reason_code TEXT,
    reason_description TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE allergies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row_hash TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    encounter_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    code_system TEXT,
    code TEXT,
    description TEXT,
    allergy_type TEXT,
    category TEXT,
    reaction_description TEXT,
    severity TEXT,
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE pending_exams (
    exam_request_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    exam_code TEXT NOT NULL,
    description TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    due_at TEXT,
    status TEXT NOT NULL CHECK (status = 'pending'),
    source TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic = 1),
    notice TEXT NOT NULL
);

CREATE INDEX idx_encounters_patient_date ON encounters(patient_id, started_at DESC);
CREATE INDEX idx_conditions_patient_date ON conditions(patient_id, started_at DESC);
CREATE INDEX idx_observations_patient_date ON observations(patient_id, observed_at DESC);
CREATE INDEX idx_medications_patient_date ON medications(patient_id, started_at DESC);
CREATE INDEX idx_procedures_patient_date ON procedures(patient_id, started_at DESC);
CREATE INDEX idx_allergies_patient_date ON allergies(patient_id, started_at DESC);
CREATE INDEX idx_pending_exams_patient_due ON pending_exams(patient_id, due_at);
"""


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)

