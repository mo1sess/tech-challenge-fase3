"""Serve the official Qwen3-8B QLoRA generator on a remote CUDA GPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from clinical_assistant.config import project_root
from clinical_assistant.finetuning.service import (
    create_inference_app,
    load_official_service_generator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    token = os.environ.get("TECHCARE_REMOTE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TECHCARE_REMOTE_TOKEN before starting the service")

    import uvicorn

    generator = load_official_service_generator(project_root(), arguments.adapter)
    app = create_inference_app(generator, token=token)
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
