"""Run the stage 8 local workflow and handle an optional human-review interrupt."""

from __future__ import annotations

import argparse
import json
import uuid

from langgraph.types import Command

from clinical_assistant.config import project_root
from clinical_assistant.graph.workflow import create_clinical_workflow


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_id", help="Identificador pseudonimizado, por exemplo PAC004")
    parser.add_argument("question", help="Pergunta a processar")
    parser.add_argument(
        "--review",
        choices=("interactive", "approve", "reject"),
        default="interactive",
        help="Como responder quando o fluxo solicitar revisão humana",
    )
    return parser.parse_args()


def _decision(mode: str) -> bool:
    if mode == "approve":
        return True
    if mode == "reject":
        return False
    answer = input("Aprovar a prévia somente para demonstração? [s/N]: ").strip().casefold()
    return answer in {"s", "sim", "y", "yes"}


if __name__ == "__main__":
    arguments = _arguments()
    graph = create_clinical_workflow(project_root())
    thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "patient_id": arguments.patient_id,
            "question": arguments.question,
            "execution_id": thread_id,
            "trace": [],
        },
        config=run_config,
    )
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(json.dumps({"status": "awaiting_human_review", "request": payload}, ensure_ascii=False, indent=2))
        result = graph.invoke(
            Command(
                resume={
                    "approved": _decision(arguments.review),
                    "reviewer": "cli_demo_user",
                    "notes": "Decisão de demonstração da ETAPA 8.",
                }
            ),
            config=run_config,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
