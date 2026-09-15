from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from model.bpe_tokenizer import VedaBPETokenizer


def prepare(input_path: Path, output_dir: Path, tokenizer_path: Path, validation_fraction: float, seed: int) -> None:
    tokenizer = VedaBPETokenizer.from_file(tokenizer_path)
    documents = []
    with input_path.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("text"):
                documents.append(item)
    random.Random(seed).shuffle(documents)
    split_index = max(1, int(len(documents) * (1.0 - validation_fraction)))
    splits = {"train": documents[:split_index], "validation": documents[split_index:]}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"source": str(input_path), "tokenizer": str(tokenizer_path), "documents": len(documents), "validation_fraction": validation_fraction, "seed": seed}
    for name, split_documents in splits.items():
        tokens: list[int] = []
        characters = 0
        for item in split_documents:
            text = item["text"]
            tokens.extend(tokenizer.encode(text))
            characters += len(text)
        array = np.asarray(tokens, dtype=np.uint16)
        array.tofile(output_dir / f"{name}.bin")
        metadata[f"{name}_documents"] = len(split_documents)
        metadata[f"{name}_characters"] = characters
        metadata[f"{name}_tokens"] = int(array.size)
        print(f"{name}: documents={len(split_documents):,} characters={characters:,} tokens={array.size:,}")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and pack Veda Pretraining Corpus v1")
    parser.add_argument("--input", type=Path, default=Path("data/pretraining-v1/documents.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pretraining-v1-tokenized"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/veda-v0.2-tokenizer-mixed.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.tokenizer, args.validation_fraction, args.seed)