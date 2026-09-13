"""Functional StateGraph with conditional routing and human review."""

from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from clinical_assistant.chains.context_chain import ContextBuilderChain
from clinical_assistant.config import load_yaml
from clinical_assistant.database.patient_repository import validate_patient_id
from clinical_assistant.graph.state import ClinicalWorkflowState
from clinical_assistant.rag.pipeline import open_protocol_retriever
from clinical_assistant.tools.langchain_tools import LangChainToolbox, create_langchain_toolbox
from clinical_assistant.tools.patient_tools import create_patient_tools


DISCLAIMER = (
    "Dados sintéticos para demonstração acadêmica. Não usar para diagnóstico, "
    "prescrição ou decisão clínica autônoma."
)


class ResponseGenerator(Protocol):
    mode: str

    def generate(self, prompt: str, state: ClinicalWorkflowState) -> str: ...


class DeterministicEvidencePreview:
    """Offline preview used to test orchestration; it is not the official LLM."""

    mode = "deterministic_evidence_preview"

    def generate(self, prompt: str, state: ClinicalWorkflowState) -> str:
        patient = state.get("patient_data", {}).get("patient")
        if not patient or patient.get("error") == "patient_not_found":
            return "Paciente pseudonimizado não localizado; nenhum dado clínico foi gerado."
        pending = state.get("pending_exams", [])
        documents = state.get("retrieved_documents", [])
        lines = [
            f"Prévia técnica para {state['patient_id']} baseada apenas nos dados recuperados.",
            f"Exames explicitamente pendentes encontrados: {len(pending)}.",
            f"Trechos de protocolo sintético recuperados: {len(documents)}.",
            "Nenhuma prescrição, diagnóstico ou alteração de tratamento foi produzida.",
            "O Qwen3-8B não foi executado neste teste local de orquestração.",
        ]
        return " ".join(lines)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _contains_any(question: str, terms: tuple[str, ...] | list[str]) -> bool:
    normalized = _normalize(question)
    return any(_normalize(term) in normalized for term in terms)


def _patient_tool_selection(question: str) -> list[str]:
    selected = ["get_patient"]
    groups = (
        (
            "get_patient_conditions",
            ("condição", "doença", "diagnóstico", "asma", "problema clínico"),
        ),
        (
            "get_patient_medications",
            ("medicamento", "medicação", "remédio", "dose", "prescrição", "tratamento"),
        ),
        (
            "get_patient_observations",
            ("observação", "resultado", "medida", "pressão", "frequência", "laboratório"),
        ),
    )
    for tool_name, terms in groups:
        if _contains_any(question, terms):
            selected.append(tool_name)
    return selected


@dataclass
class ClinicalWorkflow:
    toolbox: LangChainToolbox
    context_chain: ContextBuilderChain
    generator: ResponseGenerator
    max_question_chars: int
    review_terms: list[str]
    review_notice: str

    def __post_init__(self) -> None:
        self.tool_map = self.toolbox.by_name()

    def validate_input(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        patient_id = validate_patient_id(str(state.get("patient_id", "")))
        question = " ".join(str(state.get("question", "")).split())
        if not question:
            raise ValueError("question cannot be empty")
        if len(question) > self.max_question_chars:
            raise ValueError(f"question exceeds {self.max_question_chars} characters")
        return {
            "patient_id": patient_id,
            "question": question,
            "execution_id": state.get("execution_id") or str(uuid.uuid4()),
            "trace": ["validate_input"],
        }

    def load_patient(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        selected = _patient_tool_selection(state["question"])
        patient_data: dict[str, Any] = {}
        for tool_name in selected:
            result = self.tool_map[tool_name].invoke({"patient_id": state["patient_id"]})
            key = {
                "get_patient": "patient",
                "get_patient_conditions": "conditions",
                "get_patient_medications": "medications",
                "get_patient_observations": "observations",
            }[tool_name]
            patient_data[key] = result.get("data")
        return {
            "patient_data": patient_data,
            "selected_tools": selected,
            "trace": ["load_patient"],
        }

    def check_pending_exams(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        result = self.tool_map["get_pending_exams"].invoke(
            {"patient_id": state["patient_id"]}
        )
        return {
            "pending_exams": result.get("data", []) if result.get("ok") else [],
            "selected_tools": [*state.get("selected_tools", []), "get_pending_exams"],
            "trace": ["check_pending_exams"],
        }

    def retrieve_protocols(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        result = self.tool_map["search_internal_protocol"].invoke(
            {"query": state["question"]}
        )
        documents = result.get("documents", [])
        citations = result.get("citations", [])
        selected = [*state.get("selected_tools", []), "search_internal_protocol"]
        limitations: list[str] = []
        if _contains_any(state["question"], ("diretriz", "pcdt", "guideline", "oficial")):
            guideline = self.tool_map["search_clinical_guideline"].invoke(
                {"query": state["question"]}
            )
            selected.append("search_clinical_guideline")
            limitations.append(str(guideline["limitation"]))
        return {
            "retrieved_documents": documents,
            "citations": citations,
            "selected_tools": selected,
            "limitations": limitations,
            "trace": ["retrieve_protocols"],
        }

    def build_context(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        result = self.context_chain.invoke(dict(state))
        return {
            "context": result["context"],
            "prompt": result["prompt"],
            "trace": ["build_context"],
        }

    def generate_response(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        return {
            "llm_response": self.generator.generate(state["prompt"], state),
            "generator_mode": self.generator.mode,
            "trace": ["generate_response"],
        }

    def safety_check(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        triggers = [term for term in self.review_terms if _contains_any(state["question"], (term,))]
        requires_review = bool(triggers)
        return {
            "requires_human_validation": requires_review,
            "safety_result": {
                "status": "human_review_required" if requires_review else "informational",
                "triggers": triggers,
                "autonomous_clinical_action_allowed": False,
            },
            "trace": ["safety_check"],
        }

    @staticmethod
    def route_after_safety(state: ClinicalWorkflowState) -> Literal["human_review", "finalize_response"]:
        return "human_review" if state["requires_human_validation"] else "finalize_response"

    def human_review(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        decision = interrupt(
            {
                "execution_id": state["execution_id"],
                "patient_id": state["patient_id"],
                "question": state["question"],
                "draft": state["llm_response"],
                "citations": state.get("citations", []),
                "notice": self.review_notice,
                "instruction": "Aprovar ou rejeitar a liberação desta prévia.",
            }
        )
        if isinstance(decision, bool):
            review = {"approved": decision, "reviewer": "not_informed", "notes": ""}
        elif isinstance(decision, dict) and isinstance(decision.get("approved"), bool):
            review = {
                "approved": decision["approved"],
                "reviewer": str(decision.get("reviewer", "not_informed")),
                "notes": str(decision.get("notes", "")),
            }
        else:
            raise ValueError("human review must provide a boolean approved decision")
        return {"human_validation": review, "trace": ["human_review"]}

    def finalize_response(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        citations = state.get("citations", [])
        sources = "\n".join(f"- {citation}" for citation in citations)
        if state.get("requires_human_validation"):
            review = state.get("human_validation", {})
            if review.get("approved") is not True:
                response = (
                    "Resposta retida após revisão humana. Nenhuma orientação clínica foi liberada."
                )
            else:
                response = (
                    f"{state['llm_response']}\n\n{self.review_notice} "
                    "A aprovação registrada demonstra o fluxo e não comprova revisão médica profissional."
                )
        else:
            response = state["llm_response"]
        if sources:
            response += f"\n\nFontes recuperadas:\n{sources}"
        response += f"\n\n{DISCLAIMER}"
        return {"final_response": response, "trace": ["finalize_response"]}

    def compile(self, *, checkpointer=None):
        builder = StateGraph(ClinicalWorkflowState)
        builder.add_node("validate_input", self.validate_input)
        builder.add_node("load_patient", self.load_patient)
        builder.add_node("check_pending_exams", self.check_pending_exams)
        builder.add_node("retrieve_protocols", self.retrieve_protocols)
        builder.add_node("build_context", self.build_context)
        builder.add_node("generate_response", self.generate_response)
        builder.add_node("safety_check", self.safety_check)
        builder.add_node("human_review", self.human_review)
        builder.add_node("finalize_response", self.finalize_response)
        builder.add_edge(START, "validate_input")
        builder.add_edge("validate_input", "load_patient")
        builder.add_edge("load_patient", "check_pending_exams")
        builder.add_edge("check_pending_exams", "retrieve_protocols")
        builder.add_edge("retrieve_protocols", "build_context")
        builder.add_edge("build_context", "generate_response")
        builder.add_edge("generate_response", "safety_check")
        builder.add_conditional_edges(
            "safety_check",
            self.route_after_safety,
            {"human_review": "human_review", "finalize_response": "finalize_response"},
        )
        builder.add_edge("human_review", "finalize_response")
        builder.add_edge("finalize_response", END)
        return builder.compile(checkpointer=checkpointer or MemorySaver())


def create_clinical_workflow(root: Path, *, generator: ResponseGenerator | None = None):
    config = load_yaml(root / "configs" / "agent.yaml")
    patient_tools = create_patient_tools(root)
    retriever = open_protocol_retriever(root)
    toolbox = create_langchain_toolbox(
        patient_tools,
        retriever,
        guideline_limitation=config["context"]["unavailable_sources"][
            "search_clinical_guideline"
        ],
    )
    workflow = ClinicalWorkflow(
        toolbox=toolbox,
        context_chain=ContextBuilderChain(
            max_context_chars=int(config["execution"]["max_context_chars"])
        ),
        generator=generator or DeterministicEvidencePreview(),
        max_question_chars=int(config["execution"]["max_question_chars"]),
        review_terms=list(map(str, config["human_review"]["trigger_terms"])),
        review_notice=str(config["human_review"]["notice"]),
    )
    return workflow.compile()
