from __future__ import annotations

from types import SimpleNamespace

import pytest

from clinical_assistant.finetuning.inference import render_prompt
from clinical_assistant.finetuning.trainer import training_gpu_environment


class _NoCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _SmallCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_properties(index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name="Low VRAM test GPU", total_memory=4 * 1024**3)

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index == 0
        return (7, 5)


class _Tokenizer:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.arguments = kwargs
        return "|".join(message["content"] for message in messages)


@pytest.mark.unit
def test_training_blocks_cpu() -> None:
    torch = SimpleNamespace(cuda=_NoCuda(), version=SimpleNamespace(cuda=None))
    with pytest.raises(RuntimeError, match="remote CUDA GPU"):
        training_gpu_environment(torch, 14)


@pytest.mark.unit
def test_training_blocks_local_4gb_gpu() -> None:
    torch = SimpleNamespace(cuda=_SmallCuda(), version=SimpleNamespace(cuda="11.1"))
    with pytest.raises(RuntimeError, match="at least 14.0 GB"):
        training_gpu_environment(torch, 14)


@pytest.mark.unit
def test_prompt_disables_qwen_thinking_mode() -> None:
    tokenizer = _Tokenizer()
    prompt = render_prompt(
        tokenizer,
        system_prompt="Regras",
        instruction="Pergunta",
        input_text="Contexto",
    )
    assert prompt == "Regras|Pergunta\n\nContexto:\nContexto"
    assert tokenizer.arguments == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
