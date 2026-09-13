"""Inference helpers for the separate Qwen3-8B LoRA adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_prompt(
    tokenizer: Any,
    *,
    system_prompt: str,
    instruction: str,
    input_text: str = "",
) -> str:
    """Render the exact non-thinking chat prompt used by training and inference."""

    user_content = instruction.strip()
    if input_text.strip():
        user_content += f"\n\nContexto:\n{input_text.strip()}"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_response(
    model: Any,
    tokenizer: Any,
    torch: Any,
    *,
    prompt: str,
    max_new_tokens: int = 192,
) -> str:
    """Generate a deterministic smoke-test response from a loaded adapter."""

    model.eval()
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_tokens = generated[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


def load_adapter_for_inference(
    *,
    model_id: str,
    revision: str,
    adapter_directory: Path,
    compute_dtype: Any,
    quantization_config: Any,
    auto_model_class: Any,
    auto_tokenizer_class: Any,
    peft_model_class: Any,
) -> tuple[Any, Any]:
    """Load the immutable base model plus a separate LoRA adapter."""

    tokenizer = auto_tokenizer_class.from_pretrained(model_id, revision=revision)
    base_model = auto_model_class.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        trust_remote_code=False,
    )
    model = peft_model_class.from_pretrained(base_model, adapter_directory)
    return model, tokenizer
