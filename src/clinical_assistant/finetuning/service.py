"""GPU-only inference service for the official Qwen3-8B QLoRA adapter."""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.finetuning.inference import (
    generate_response,
    load_adapter_for_inference,
    render_prompt,
)


OFFICIAL_GENERATOR_MODE = "qwen3_8b_qlora_remote"


def validate_official_adapter(root: Path, adapter_directory: Path) -> dict[str, Any]:
    """Validate the final adapter against the immutable stage-5 manifest."""

    adapter_directory = adapter_directory.resolve()
    weights = adapter_directory / "adapter_model.safetensors"
    adapter_config = adapter_directory / "adapter_config.json"
    if not weights.is_file() or not adapter_config.is_file():
        raise FileNotFoundError(
            "The adapter directory must contain adapter_model.safetensors and adapter_config.json"
        )

    evaluation_config = load_yaml(root / "configs" / "model_evaluation.yaml")
    manifest_path = resolve_project_path(
        evaluation_config["prerequisites"]["training_manifest"], root=root
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = str(
        manifest.get("artifact_sha256", {}).get("adapter/adapter_model.safetensors", "")
    )
    actual_hash = sha256_file(weights)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"Adapter SHA-256 mismatch: expected {expected_hash}, found {actual_hash}"
        )
    model = manifest.get("model", {})
    if model.get("id") != "Qwen/Qwen3-8B":
        raise ValueError("The stage-5 manifest does not identify the official Qwen3-8B")
    return {
        "adapter_directory": str(adapter_directory),
        "adapter_sha256": actual_hash,
        "model_id": str(model["id"]),
        "revision": str(model["revision"]),
        "training_manifest": str(manifest_path),
    }


def inference_gpu_environment(torch: Any, minimum_vram_gb: float) -> dict[str, Any]:
    """Reject CPU and the local 4 GB GPU before model loading begins."""

    if not torch.cuda.is_available():
        raise RuntimeError("Official agent inference requires a remote CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    total_gb = properties.total_memory / (1024**3)
    if total_gb < minimum_vram_gb:
        raise RuntimeError(
            f"GPU has {total_gb:.1f} GB VRAM; at least {minimum_vram_gb:.1f} GB is required"
        )
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu_name": properties.name,
        "gpu_vram_gb": total_gb,
        "compute_capability": f"{major}.{minor}",
        "cuda_runtime": torch.version.cuda,
    }


@dataclass
class QwenAdapterServiceGenerator:
    model: Any
    tokenizer: Any
    torch: Any
    model_metadata: dict[str, str]
    system_prompt: str
    max_new_tokens: int
    environment: dict[str, Any]
    mode: str = OFFICIAL_GENERATOR_MODE

    def generate(self, prompt: str, state: dict[str, Any]) -> str:
        del state
        rendered = render_prompt(
            self.tokenizer,
            system_prompt=self.system_prompt,
            instruction=prompt,
        )
        return generate_response(
            self.model,
            self.tokenizer,
            self.torch,
            prompt=rendered,
            max_new_tokens=self.max_new_tokens,
        )


def load_official_service_generator(
    root: Path, adapter_directory: Path
) -> QwenAdapterServiceGenerator:
    """Load the pinned 4-bit base model and verified final adapter on CUDA."""

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements/full-agent-gpu.txt in a Linux CUDA runtime"
        ) from exc

    config = load_yaml(root / "configs" / "full_agent.yaml")
    adapter = validate_official_adapter(root, adapter_directory)
    environment = inference_gpu_environment(
        torch, float(config["execution"]["minimum_vram_gb"])
    )
    quantization = config["quantization"]
    compute_dtype = torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["quant_type"]),
        bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model, tokenizer = load_adapter_for_inference(
        model_id=adapter["model_id"],
        revision=adapter["revision"],
        adapter_directory=Path(adapter["adapter_directory"]),
        compute_dtype=compute_dtype,
        quantization_config=quantization_config,
        auto_model_class=AutoModelForCausalLM,
        auto_tokenizer_class=AutoTokenizer,
        peft_model_class=PeftModel,
    )
    model.config.use_cache = True
    return QwenAdapterServiceGenerator(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        model_metadata={
            "id": adapter["model_id"],
            "revision": adapter["revision"],
            "adapter_sha256": adapter["adapter_sha256"],
        },
        system_prompt=str(config["generation"]["system_prompt"]),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        environment=environment,
    )


def create_inference_app(generator: QwenAdapterServiceGenerator, *, token: str):
    """Create a token-protected FastAPI app without importing it locally."""

    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install requirements/full-agent-gpu.txt") from exc
    token = token.strip()
    if len(token) < 24:
        raise ValueError("The inference token must contain at least 24 characters")

    app = FastAPI(title="TechCare Qwen3-8B QLoRA", docs_url=None, redoc_url=None)

    def authorize(authorization: str | None) -> None:
        expected = f"Bearer {token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        return {
            "ok": True,
            "generator_mode": generator.mode,
            "model": generator.model_metadata,
            "environment": generator.environment,
        }

    @app.post("/v1/generate")
    def generate(
        payload: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(authorization)
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="prompt is required")
        if len(prompt) > 20000:
            raise HTTPException(status_code=422, detail="prompt exceeds 20000 characters")
        response = generator.generate(prompt, {})
        return {
            "response": response,
            "generator_mode": generator.mode,
            "model": generator.model_metadata,
        }

    return app
