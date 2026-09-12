from __future__ import annotations

import torch

from evaluation.perplexity import load_model
from model.bpe_tokenizer import VedaBPETokenizer
from model.tokenizer import ByteTokenizer


PROMPTS = ["Once upon a time", "The little girl", "A dog went into", "One morning", "The robot"]


def generate(checkpoint: str, prompt: str, tokens: int = 80, tokenizer_path: str | None = None) -> str:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(checkpoint, device)
    tokenizer = VedaBPETokenizer.from_file(tokenizer_path) if tokenizer_path else ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(input_ids[:, -model.config.context_length :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == tokenizer.eos_id:
                break
    return tokenizer.decode(input_ids[0].tolist())