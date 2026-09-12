from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def take_texts(dataset: Iterable[dict], field: str, source: str, limit: int):
    total = 0
    documents = 0
    for row in dataset:
        value = row.get(field, "")
        if isinstance(value, list):
            value = "\n".join(message.get("content", "") for message in value if isinstance(message, dict))
        text = str(value).strip()
        if not text:
            continue
        remaining = limit - total
        text = text[:remaining]
        yield {"source": source, "text": text}
        total += len(text)
        documents += 1
        if total >= limit:
            break
    print(f"{source}: documents={documents:,} characters={total:,}")


def build(output: Path) -> None:
    sources = [
        ("roneneldan/TinyStories", "default", "text", "general", 5_000_000),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", "technical", 5_000_000),
        ("open-web-math/open-web-math", "default", "text", "math", 2_000_000),
        ("HuggingFaceTB/smoltalk", "all", "messages", "code-technical", 3_000_000),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for name, config, field, category, limit in sources:
            dataset = load_dataset(name, config, split="train", streaming=True)
            for item in take_texts(dataset, field, f"{name}:{category}", limit):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a provenance-tracked mixed corpus for the Veda BPE tokenizer")
    parser.add_argument("--output", type=Path, default=Path("data/tokenizer_corpus/mixed.jsonl"))
    args = parser.parse_args()
    build(args.output)