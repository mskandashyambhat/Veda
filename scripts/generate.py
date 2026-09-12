from __future__ import annotations

import argparse

import torch

from model.config import VedaConfig
from model.bpe_tokenizer import VedaBPETokenizer
from model.tokenizer import ByteTokenizer
from model.veda import VedaModel


def generate(checkpoint: str, prompt: str, tokens_to_generate: int, tokenizer_path: str | None = None) -> str:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config = VedaConfig(**saved["config"])
    model = VedaModel(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    tokenizer = VedaBPETokenizer.from_file(tokenizer_path) if tokenizer_path else ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(tokens_to_generate):
            context = input_ids[:, -config.context_length :]
            logits, _ = model(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == tokenizer.eos_id:
                break
    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/veda-10m/veda.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    print(generate(args.checkpoint, args.prompt, args.tokens, args.tokenizer))