"""State shared by all stage 8 workflow nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ClinicalWorkflowState(TypedDict, total=False):
    patient_id: str
    question: str
    patient_data: dict[str, Any]
    pending_exams: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]
    citations: list[str]
    context: str
    prompt: str
    llm_response: str
    input_safety_result: dict[str, Any]
    safety_result: dict[str, Any]
    requires_human_validation: bool
    human_validation: dict[str, Any]
    final_response: str
    audit_result: dict[str, Any]
    execution_id: str
    generator_mode: str
    selected_tools: list[str]
    limitations: list[str]
    trace: Annotated[list[str], operator.add]
