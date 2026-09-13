"""Run a local protocol retrieval and print source-aware JSON results."""

from __future__ import annotations

import argparse
import json

from clinical_assistant.config import project_root
from clinical_assistant.rag.pipeline import open_protocol_retriever


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Pergunta usada para recuperar protocolos")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    hits = open_protocol_retriever(project_root()).retrieve(arguments.query)
    result = {
        "query": arguments.query,
        "results": [hit.to_dict() for hit in hits],
        "sources": [hit.citation for hit in hits],
        "warning": (
            "Base sintética para demonstração acadêmica. "
            "Não usar como orientação clínica; validação médica necessária."
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

