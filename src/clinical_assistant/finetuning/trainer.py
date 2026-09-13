"""Portable remote-GPU QLoRA trainer for the pinned Qwen3-8B model."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.common import sha256_file, utc_now, write_json
from clinical_assistant.config import load_yaml, resolve_project_path
from clinical_assistant.finetuning.dataset import validate_finetuning_dataset
from clinical_assistant.finetuning.inference import generate_response, render_prompt


def _import_training_stack() -> dict[str, Any]:
    try:
        import accelerate
        import bitsandbytes
        import datasets
        import peft
        import torch
        import transformers
        import trl
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements/gpu-colab-kaggle.txt in a Linux CUDA runtime"
        ) from exc
    return {
        "accelerate": accelerate,
        "bitsandbytes": bitsandbytes,
        "datasets": datasets,
        "peft": peft,
        "torch": torch,
        "transformers": transformers,
        "trl": trl,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def training_gpu_environment(torch: Any, minimum_vram_gb: float) -> dict[str, Any]:
    """Require a remote CUDA GPU and reject the 4 GB local device."""

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a remote CUDA GPU")
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


def _create_run_directory(base: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("training-%Y%m%dT%H%M%SZ")
    destination = base / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _trainable_parameter_counts(model: Any) -> dict[str, Any]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100 * trainable / total,
    }


def _render_for_training(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    prompt = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    completion = str(example["completion"][0]["content"]).strip() + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


def _write_log_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for entry in history:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def run_qlora_training(root: Path) -> dict[str, Any]:
    """Train, validate and save a separate QLoRA adapter with real evidence."""

    config_path = root / "configs" / "training.yaml"
    config = load_yaml(config_path)
    model_config = config["model"]
    if model_config["id"] != "Qwen/Qwen3-8B" or model_config["allow_substitution"]:
        raise ValueError("The official Qwen3-8B model cannot be substituted")
    if not model_config["separate_adapter"]:
        raise ValueError("The LoRA adapter must remain separate from the base model")

    dataset_validation = validate_finetuning_dataset(root)
    if not dataset_validation["ok"]:
        raise ValueError(f"Invalid fine-tuning dataset: {dataset_validation['errors']}")

    stack = _import_training_stack()
    torch = stack["torch"]
    execution = config["execution"]
    environment = training_gpu_environment(torch, float(execution["minimum_vram_gb"]))
    seed = int(config["reproducibility"]["seed"])
    torch.manual_seed(seed)
    stack["transformers"].set_seed(seed)

    output_base = resolve_project_path(execution["output_directory"], root=root)
    run_directory = _create_run_directory(output_base)
    write_json(
        run_directory / "run_status.json",
        {"stage": 5, "complete": False, "started_at_utc": utc_now()},
    )

    quantization = config["quantization"]
    compute_dtype = torch.float16
    quantization_config = stack["BitsAndBytesConfig"](
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["quant_type"]),
        bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_id = str(model_config["id"])
    revision = str(model_config["revision"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(model_id, revision=revision)
    model = stack["AutoModelForCausalLM"].from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        trust_remote_code=bool(model_config["trust_remote_code"]),
    )
    model.config.use_cache = False
    lora = config["lora"]
    peft_config = stack["LoraConfig"](
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        task_type="CAUSAL_LM",
        target_modules=list(map(str, lora["target_modules"])),
    )

    dataset_directory = resolve_project_path(config["dataset"]["output_directory"], root=root)
    data_files = {
        split: str(dataset_directory / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    loaded = stack["load_dataset"]("json", data_files=data_files)
    original_columns = loaded["train"].column_names
    rendered = loaded.map(
        lambda example: _render_for_training(example, tokenizer),
        remove_columns=original_columns,
        desc="Rendering Qwen3 non-thinking prompts",
    )

    training = config["training"]
    training_args = stack["SFTConfig"](
        output_dir=str(run_directory / "checkpoints"),
        max_length=int(training["max_length"]),
        packing=bool(training["packing"]),
        completion_only_loss=True,
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        warmup_ratio=float(training["warmup_ratio"]),
        optim=str(training["optim"]),
        max_grad_norm=float(training["max_grad_norm"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy=str(training["eval_strategy"]),
        save_strategy=str(training["save_strategy"]),
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        eos_token=tokenizer.eos_token,
    )
    trainer = stack["SFTTrainer"](
        model=model,
        args=training_args,
        train_dataset=rendered["train"],
        eval_dataset=rendered["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    parameter_counts = _trainable_parameter_counts(trainer.model)
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_directory = run_directory / "adapter"
    trainer.model.save_pretrained(adapter_directory, safe_serialization=True)
    tokenizer.save_pretrained(adapter_directory)
    trainer.save_state()
    write_json(run_directory / "train_metrics.json", dict(train_result.metrics))
    write_json(run_directory / "validation_metrics.json", dict(eval_metrics))
    _write_log_history(run_directory / "log_history.jsonl", trainer.state.log_history)

    trainer.model.config.use_cache = True
    smoke_instruction = str(config["smoke_test"]["instruction"])
    smoke_input = str(config["smoke_test"]["input"])
    smoke_prompt = render_prompt(
        tokenizer,
        system_prompt=str(config["dataset"]["system_prompt"]),
        instruction=smoke_instruction,
        input_text=smoke_input,
    )
    smoke_response = generate_response(
        trainer.model,
        tokenizer,
        torch,
        prompt=smoke_prompt,
        max_new_tokens=int(config["smoke_test"]["max_new_tokens"]),
    )
    write_json(
        run_directory / "inference_smoke.json",
        {
            "instruction": smoke_instruction,
            "input": smoke_input,
            "response": smoke_response,
            "generated_at_utc": utc_now(),
        },
    )

    write_json(
        run_directory / "run_status.json",
        {"stage": 5, "complete": True, "completed_at_utc": utc_now()},
    )
    manifest = {
        "schema_version": 1,
        "stage": 5,
        "run_type": "official_qlora_training",
        "complete": True,
        "generated_at_utc": utc_now(),
        "model": {"id": model_id, "revision": revision, "adapter": "adapter"},
        "quantization": {**quantization, "effective_compute_dtype": str(compute_dtype)},
        "lora": lora,
        "training": training,
        "dataset": dataset_validation,
        "trainable_parameters": parameter_counts,
        "metrics": {
            "train": dict(train_result.metrics),
            "validation": dict(eval_metrics),
        },
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "environment": {
            **environment,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": stack["transformers"].__version__,
            "peft": stack["peft"].__version__,
            "trl": stack["trl"].__version__,
            "accelerate": stack["accelerate"].__version__,
            "bitsandbytes": stack["bitsandbytes"].__version__,
            "datasets": stack["datasets"].__version__,
        },
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(
            resolve_project_path(config["dataset"]["manifest"], root=root)
        ),
        "limitations": [
            "Fine-tuning on synthetic academic hospital data is not clinical validation.",
            "The adapter requires the exact pinned Qwen3-8B base revision.",
            "Comparison against the baseline is reserved for a later evaluation stage.",
        ],
    }
    manifest["artifact_sha256"] = _artifact_hashes(run_directory)
    write_json(run_directory / "run_manifest.json", manifest)
    return {"run_directory": str(run_directory), **manifest}
