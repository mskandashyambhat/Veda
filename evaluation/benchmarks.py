from __future__ import annotations

from evaluation.generation import PROMPTS, generate


def run_generation_benchmark(checkpoint: str, tokens: int = 80, tokenizer_path: str | None = None) -> list[dict[str, str]]:
    return [{"prompt": prompt, "output": generate(checkpoint, prompt, tokens, tokenizer_path)} for prompt in PROMPTS]