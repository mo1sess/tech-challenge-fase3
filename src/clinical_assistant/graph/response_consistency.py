"""Evidence locks for factual patient responses produced by a remote LLM."""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Any


PLACEHOLDER_PATTERN = re.compile(r"\[[^\]\n]{2,48}\]")
LOOKUP_TERMS = (
    "qual",
    "quais",
    "liste",
    "listar",
    "mostre",
    "mostrar",
    "aparece",
    "aparecem",
    "registrado",
    "registrada",
    "registrados",
    "registradas",
    "consta",
    "constam",
    "possui",
    "tem",
)
ACTION_TERMS = (
    "alterar",
    "tomar",
    "usar",
    "prescrever",
    "receitar",
    "iniciar",
    "interromper",
    "suspender",
    "dose correta",
)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalize(value)
    return any(_normalize(term) in normalized for term in terms)


def _clean(value: Any, *, missing: str = "não informado") -> str:
    cleaned = " ".join(str(value or "").split())
    return cleaned or missing


def _lookup_requested(question: str) -> bool:
    return _contains_any(question, LOOKUP_TERMS) and not _contains_any(
        question, ACTION_TERMS
    )


def factual_evidence_kind(question: str) -> str | None:
    """Classify bounded read-only patient questions that require exact values."""

    if _contains_any(question, ("exame", "exames")) and _contains_any(
        question, ("pendente", "pendentes", "solicitação", "solicitacao")
    ):
        return "pending_exams"
    if not _lookup_requested(question):
        return None
    if _contains_any(
        question,
        ("medicamento", "medicamentos", "medicação", "medicacao", "remédio", "remedio"),
    ):
        return "medications"
    if _contains_any(
        question,
        ("condição", "condicao", "condições", "condicoes", "doença", "doenca", "diagnóstico", "diagnostico"),
    ):
        return "conditions"
    if _contains_any(
        question,
        ("observação", "observacao", "observações", "observacoes", "resultado", "resultados", "medida", "pressão", "pressao", "frequência", "frequencia", "laboratório", "laboratorio"),
    ):
        return "observations"
    return None


def _medication_response(patient_id: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return f"Nenhum medicamento foi encontrado no prontuário sintético de {patient_id}."

    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for record in records:
        code = _clean(record.get("code"))
        description = _clean(record.get("description"))
        key = (code, description)
        item = grouped.setdefault(key, {"count": 0, "reasons": []})
        item["count"] += 1
        reason = _clean(record.get("reason_description"), missing="")
        if reason and reason not in item["reasons"]:
            item["reasons"].append(reason)

    lines = [f"Medicamentos registrados no prontuário sintético de {patient_id}:"]
    for (code, description), item in grouped.items():
        count = int(item["count"])
        occurrences = f"{count} registro" if count == 1 else f"{count} registros"
        reasons = "; ".join(item["reasons"]) or "não informado"
        lines.append(
            f"- {description} — código {code}; {occurrences}; "
            f"motivo registrado: {reasons}."
        )
    lines.append(
        "Nomes, códigos e motivos foram reproduzidos literalmente do SQLite; "
        "nenhuma prescrição ou recomendação foi gerada."
    )
    return "\n".join(lines)


def _condition_response(patient_id: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return f"Nenhuma condição foi encontrada no prontuário sintético de {patient_id}."
    lines = [f"Condições registradas no prontuário sintético de {patient_id}:"]
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            _clean(record.get("code_system")),
            _clean(record.get("code")),
            _clean(record.get("description")),
        )
        if key in seen:
            continue
        seen.add(key)
        system, code, description = key
        lines.append(f"- {description} — código {system} {code}.")
    lines.append(
        "As descrições e os códigos foram reproduzidos literalmente do SQLite; "
        "nenhum diagnóstico novo foi inferido."
    )
    return "\n".join(lines)


def _observation_response(patient_id: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return f"Nenhuma observação foi encontrada no prontuário sintético de {patient_id}."
    lines = [f"Observações registradas no prontuário sintético de {patient_id}:"]
    for record in records:
        description = _clean(record.get("description"))
        value = _clean(record.get("value"))
        units = _clean(record.get("units"), missing="")
        observed_at = _clean(record.get("observed_at"))
        code = _clean(record.get("code"))
        measurement = " ".join(part for part in (value, units) if part)
        lines.append(
            f"- {description}: {measurement} — código {code}; data {observed_at}."
        )
    lines.append(
        "Valores, unidades, códigos e datas foram reproduzidos literalmente do SQLite; "
        "nenhuma interpretação clínica foi realizada."
    )
    return "\n".join(lines)


def _pending_exam_response(patient_id: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return (
            f"Nenhum exame explicitamente pendente foi encontrado nos registros "
            f"sintéticos de {patient_id}."
        )
    lines = [f"Exames explicitamente pendentes para {patient_id}:"]
    for record in records:
        lines.append(
            "- {description} — solicitação {request_id}; código {code}; "
            "solicitado em {requested}; prazo {due}; estado {status}; fonte {source}.".format(
                description=_clean(record.get("description")),
                request_id=_clean(record.get("exam_request_id")),
                code=_clean(record.get("exam_code")),
                requested=_clean(record.get("requested_at")),
                due=_clean(record.get("due_at")),
                status=_clean(record.get("status")),
                source=_clean(record.get("source")),
            )
        )
    lines.append(
        "Os campos foram reproduzidos literalmente do SQLite; nenhum novo exame foi sugerido."
    )
    return "\n".join(lines)


def _records_for_kind(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "pending_exams":
        value = state.get("pending_exams", [])
    else:
        value = state.get("patient_data", {}).get(kind, [])
    return [dict(item) for item in (value or []) if isinstance(item, dict)]


def _fallback_response(state: dict[str, Any]) -> str:
    patient_id = _clean(state.get("patient_id"))
    pending = len(state.get("pending_exams", []) or [])
    documents = len(state.get("retrieved_documents", []) or [])
    return (
        f"Não foi possível liberar a redação gerada para {patient_id} porque ela continha "
        "campos não preenchidos. As evidências estruturadas indicam "
        f"{pending} exame(s) explicitamente pendente(s) e {documents} trecho(s) "
        "de protocolo sintético recuperado(s)."
    )


def enforce_response_consistency(
    candidate: str, state: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Replace unsafe-to-trust prose with exact structured patient evidence."""

    kind = factual_evidence_kind(str(state.get("question", "")))
    placeholder_detected = bool(PLACEHOLDER_PATTERN.search(candidate))
    if kind:
        records = _records_for_kind(state, kind)
        formatter = {
            "medications": _medication_response,
            "conditions": _condition_response,
            "observations": _observation_response,
            "pending_exams": _pending_exam_response,
        }[kind]
        response = formatter(str(state.get("patient_id", "")), records)
        return response, {
            "policy": "structured_patient_evidence_v1",
            "applied": True,
            "reason": f"factual_{kind}",
            "evidence_kind": kind,
            "evidence_records": records,
            "model_placeholders_detected": placeholder_detected,
        }
    if placeholder_detected:
        return _fallback_response(state), {
            "policy": "structured_patient_evidence_v1",
            "applied": True,
            "reason": "unfilled_model_placeholder",
            "evidence_kind": "summary",
            "evidence_records": [],
            "model_placeholders_detected": True,
        }
    return candidate, {
        "policy": "structured_patient_evidence_v1",
        "applied": False,
        "reason": "model_response_retained",
        "evidence_kind": "",
        "evidence_records": [],
        "model_placeholders_detected": False,
    }
