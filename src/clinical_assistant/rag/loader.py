"""Load and deduplicate the synthetic internal protocol corpus."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from clinical_assistant.rag.models import RagDocument


REQUIRED_METADATA = {
    "document_id",
    "document_type",
    "version",
    "section",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Protocol source not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object on line {line_number} of {path}")
        records.append(value)
    return records


def load_synthetic_protocols(
    path: Path,
    *,
    required_notice: str,
    expected_records: int | None = None,
) -> list[RagDocument]:
    """Return one logical document per protocol/section, removing prompt variants."""

    records = _read_jsonl(path)
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(
            f"Expected {expected_records} protocol records, found {len(records)}"
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records, 1):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or not REQUIRED_METADATA <= metadata.keys():
            raise ValueError(f"Protocol record {index} has incomplete metadata")
        if record.get("synthetic") is not True:
            raise ValueError(f"Protocol record {index} is not explicitly synthetic")
        if record.get("notice") != required_notice:
            raise ValueError(f"Protocol record {index} has an invalid academic notice")
        if metadata["document_type"] != "synthetic_internal_protocol":
            raise ValueError(f"Protocol record {index} has an invalid document type")
        output = str(record.get("output", "")).strip()
        if not output:
            raise ValueError(f"Protocol record {index} has no content")
        key = (
            str(metadata["document_id"]),
            str(metadata["version"]),
            str(metadata["section"]),
            output,
        )
        grouped[key].append(record)

    documents: list[RagDocument] = []
    for (document_id, version, section, output), variants in sorted(grouped.items()):
        instructions = sorted({str(item["instruction"]).strip() for item in variants})
        contexts = sorted({str(item.get("input", "")).strip() for item in variants if item.get("input")})
        source = str(variants[0]["source"])
        name = f"Protocolo Interno Sintético {document_id}"
        text_parts = [
            f"Documento: {name}",
            f"Seção: {section}",
            f"Assuntos: {' | '.join(instructions)}",
        ]
        if contexts:
            text_parts.append(f"Contexto: {' | '.join(contexts)}")
        text_parts.append(f"Orientação: {output}")
        text_parts.append(required_notice)
        documents.append(
            RagDocument(
                document_id=document_id,
                document_name=name,
                version=version,
                section=section,
                source=source,
                document_type="synthetic_internal_protocol",
                text="\n".join(text_parts),
                synthetic=True,
                notice=required_notice,
            )
        )
    return documents

