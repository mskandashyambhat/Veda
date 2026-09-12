from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from model.bpe_tokenizer import VedaBPETokenizer
from model.tokenizer import ByteTokenizer


SAMPLES = [
    "def calculate_total(items):",
    "नमस्ते",
    "こんにちは",
    "π ≈ 3.14159",
    "SELECT * FROM users;",
    '{"name": "Veda", "tokens": 32000}',
    "# Markdown heading\n\n- item",
]


def build_report(corpus: Path, tokenizer_path: Path) -> dict:
    tokenizer = VedaBPETokenizer.from_file(tokenizer_path)
    byte_tokenizer = ByteTokenizer()
    source_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "characters": 0, "bpe_tokens": 0, "byte_tokens": 0})
    with corpus.open() as handle:
        for line in handle:
            item = json.loads(line)
            text = item["text"]
            stats = source_stats[item["source"]]
            stats["documents"] += 1
            stats["characters"] += len(text)
            stats["bpe_tokens"] += len(tokenizer.encode(text, add_special_tokens=False))
            stats["byte_tokens"] += len(byte_tokenizer.encode(text, add_special_tokens=False))
    samples = []
    for text in SAMPLES:
        tokens = tokenizer.encode(text)
        samples.append({"text": text, "bpe_tokens": len(tokens), "byte_tokens": len(byte_tokenizer.encode(text)), "round_trip": tokenizer.decode(tokens) == text})
    return {"tokenizer": str(tokenizer_path), "vocab_size": tokenizer.tokenizer.get_vocab_size(), "sources": dict(source_stats), "samples": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report Veda BPE coverage and compression")
    parser.add_argument("--corpus", type=Path, default=Path("data/tokenizer_corpus/mixed.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/veda-v0.2-tokenizer-mixed.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/tokenizer-v0.2.json"))
    args = parser.parse_args()
    report = build_report(args.corpus, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))