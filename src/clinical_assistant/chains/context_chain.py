"""Build a compact, evidence-bound prompt with LangChain runnables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


PROMPT_TEMPLATE = """Você é um assistente acadêmico de apoio clínico.
Use exclusivamente o contexto abaixo. Não invente dados, não prescreva, não
altere medicamentos e não afirme diagnóstico definitivo. Quando a pergunta
envolver conduta clínica, indique que validação médica é necessária.

Paciente pseudonimizado: {patient_id}
Pergunta: {question}

CONTEXTO MÍNIMO:
{context}

FONTES:
{sources}

Responda de forma objetiva e diferencie dados do paciente, protocolos
sintéticos e limitações de fonte.
"""


@dataclass(frozen=True)
class ContextBuilderChain:
    max_context_chars: int = 12000

    def __post_init__(self) -> None:
        if self.max_context_chars < 1000:
            raise ValueError("max_context_chars must be at least 1000")
        object.__setattr__(
            self,
            "runnable",
            RunnableLambda(self._prepare) | PromptTemplate.from_template(PROMPT_TEMPLATE),
        )

    def _prepare(self, values: dict[str, Any]) -> dict[str, str]:
        compact = {
            "patient": values.get("patient_data", {}),
            "pending_exams": values.get("pending_exams", []),
            "retrieved_documents": values.get("retrieved_documents", []),
        }
        context = json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n[CONTEXTO TRUNCADO PELO LIMITE]"
        citations = values.get("citations", [])
        sources = "\n".join(f"- {citation}" for citation in citations)
        if not sources:
            sources = "- Nenhuma fonte clínica oficial recuperada."
        return {
            "patient_id": str(values["patient_id"]),
            "question": str(values["question"]),
            "context": context,
            "sources": sources,
        }

    def invoke(self, values: dict[str, Any]) -> dict[str, str]:
        prompt_value = self.runnable.invoke(values)
        return {
            "context": self._prepare(values)["context"],
            "prompt": prompt_value.to_string(),
        }
