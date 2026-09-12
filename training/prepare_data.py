from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from model.bpe_tokenizer import VedaBPETokenizer
from model.tokenizer import ByteTokenizer


def collect_split(split: str, max_characters: int) -> str:
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    chunks: list[str] = []
    total = 0
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        chunks.append(text)
        total += len(text)
        if total >= max_characters:
            break
    return "\n\n".join(chunks)


def prepare(output_dir: Path, max_characters: int, tokenizer_path: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = VedaBPETokenizer.from_file(tokenizer_path) if tokenizer_path else ByteTokenizer()
    tokenizer_name = str(tokenizer_path) if tokenizer_path else "byte"
    metadata = {"vocab_size": tokenizer.tokenizer.get_vocab_size() if tokenizer_path else tokenizer.vocab_size, "tokenizer": tokenizer_name, "source": "roneneldan/TinyStories"}
    for split in ("train", "validation"):
        text = collect_split(split, max_characters if split == "train" else max_characters // 10)
        tokens = np.asarray(tokenizer.encode(text), dtype=np.uint16)
        tokens.tofile(output_dir / f"{split}.bin")
        metadata[f"{split}_tokens"] = int(tokens.size)
        print(f"{split}: {tokens.size:,} tokens")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and tokenize a bounded TinyStories subset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--max-characters", type=int, default=20_000_000)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    prepare(args.output_dir, args.max_characters, args.tokenizer)