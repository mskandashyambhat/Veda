"""A deterministic byte tokenizer for the first Veda training run."""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 258
    bos_id = 256
    eos_id = 257

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = list(text.encode("utf-8", errors="replace"))
        if add_special_tokens:
            return [self.bos_id, *tokens, self.eos_id]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_values = [token for token in tokens if 0 <= token <= 255]
        return bytes(byte_values).decode("utf-8", errors="replace")