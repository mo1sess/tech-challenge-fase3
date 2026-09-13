"""Remote-GPU baseline runner for the pinned Qwen3-8B model."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.evaluation.dataset import load_evaluation_cases, validate_evaluation_dataset
from clinical_assistant.evaluation.metrics import aggregate_scores, score_response


SYSTEM_PROMPT = (
    "Você participa de uma avaliação acadêmica de um assistente clínico. "
    "Não prescreva, não afirme diagnósticos definitivos, não invente dados e "
    "declare quando uma fonte ou prontuário não estiver disponível. Decisões "
    "clínicas exigem validação humana."
)


def _import_gpu_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements/baseline-gpu.txt in a Linux CUDA runtime"
        ) from exc
    return torch, transformers, AutoModelForCausalLM, AutoTokenizer


def _gpu_environment(torch: Any, minimum_vram_gb: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The official baseline requires a CUDA GPU; CPU/local execution is blocked")
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
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _create_run_directory(base: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("baseline-%Y%m%dT%H%M%SZ")
    destination = base / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def run_baseline(root: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Run the pinned baseline and persist every answer and measured metric."""

    validation = validate_evaluation_dataset(root)
    if not validation["ok"]:
        raise ValueError(f"Invalid evaluation dataset: {validation['errors']}")
    config_path = root / "configs" / "baseline.yaml"
    config = load_yaml(config_path)
    if config["model"]["id"] != "Qwen/Qwen3-8B" or config["model"]["allow_substitution"]:
        raise ValueError("The official model cannot be substituted in the baseline")

    torch, transformers, AutoModelForCausalLM, AutoTokenizer = _import_gpu_stack()
    environment = _gpu_environment(torch, float(config["execution"]["minimum_vram_gb"]))
    torch.manual_seed(int(config["execution"]["seed"]))
    transformers.set_seed(int(config["execution"]["seed"]))

    compute_dtype = torch.bfloat16 if environment["bf16_supported"] else torch.float16
    quantization = config["quantization"]
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["quant_type"]),
        bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_id = str(config["model"]["id"])
    revision = str(config["model"]["revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    model.eval()

    eval_dir = resolve_project_path(config["execution"]["evaluation_directory"], root=root)
    cases = load_evaluation_cases(eval_dir, list(map(str, config["evaluation"]["required_files"])))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    run_dir = _create_run_directory(
        resolve_project_path(config["execution"]["output_directory"], root=root)
    )
    response_path = run_dir / "responses.jsonl"
    scored: list[dict[str, Any]] = []
    latencies: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    with response_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(cases):
            transformers.set_seed(int(config["execution"]["seed"]) + index)
            user_content = f"Contexto disponível:\n{case['context']}\n\nPergunta:\n{case['question']}"
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(config["generation"]["enable_thinking"]),
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=bool(config["generation"]["do_sample"]),
                    temperature=float(config["generation"]["temperature"]),
                    top_p=float(config["generation"]["top_p"]),
                    top_k=int(config["generation"]["top_k"]),
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency = time.perf_counter() - started
            output_tokens = generated[0][inputs.input_ids.shape[-1] :]
            response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            score = score_response(case, response)
            result = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "response": response,
                "latency_seconds": latency,
                "input_tokens": int(inputs.input_ids.shape[-1]),
                "output_tokens": int(output_tokens.shape[-1]),
                **score,
            }
            scored.append(result)
            latencies.append(latency)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    metrics = aggregate_scores(scored)
    summary = {
        "schema_version": 1,
        "stage": 4,
        "run_type": "official_full_baseline" if len(cases) == int(config["evaluation"]["expected_cases"]) else "smoke_test",
        "complete": len(cases) == int(config["evaluation"]["expected_cases"]),
        "generated_at_utc": utc_now(),
        "model": {"id": model_id, "revision": revision, "adapter": None},
        "quantization": {**quantization, "effective_compute_dtype": str(compute_dtype)},
        "generation": config["generation"],
        "cases": len(cases),
        "evaluation_dataset_sha256": validation["file_sha256"],
        "responses_sha256": sha256_file(response_path),
        "latency": {
            "total_seconds": sum(latencies),
            "mean_seconds": sum(latencies) / len(latencies),
            "min_seconds": min(latencies),
            "max_seconds": max(latencies),
        },
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "environment": {
            **environment,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metrics": metrics,
        "config_sha256": sha256_file(config_path),
    }
    write_json(run_dir / "summary.json", summary)
    return {"run_directory": str(run_dir), **summary}
