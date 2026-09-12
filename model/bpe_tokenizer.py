from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class VedaBPETokenizer:
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "<|system|>", "<|tool|>", "<|tool_result|>"]

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    @classmethod
    def from_file(cls, path: str | Path) -> "VedaBPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        if add_special_tokens:
            return [self.bos_id, *tokens, self.eos_id]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode([token for token in tokens if token not in {self.bos_id, self.eos_id, self.pad_id}])