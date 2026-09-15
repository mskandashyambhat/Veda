from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def extract_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(message.get("content", "") for message in value if isinstance(message, dict))
    return str(value)


def collect(dataset: Iterable[dict], field: str, source: str, category: str, limit: int, seen: set[str]):
    characters = 0
    documents = 0
    duplicates = 0
    for row in dataset:
        text = normalize(extract_text(row.get(field, "")))
        if len(text) < 80:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        remaining = limit - characters
        if remaining <= 0:
            break
        text = text[:remaining]
        seen.add(digest)
        yield {"text": text, "source": source, "category": category, "sha256": digest}
        characters += len(text)
        documents += 1
    print(f"{source}: documents={documents:,} characters={characters:,} duplicates={duplicates:,}")


def build(output: Path) -> None:
    sources = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", "general-educational", 10_000_000),
        ("open-web-math/open-web-math", "default", "text", "mathematics", 5_000_000),
        ("HuggingFaceTB/smoltalk", "all", "messages", "code-technical", 5_000_000),
        ("roneneldan/TinyStories", "default", "text", "general-story", 5_000_000),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with output.open("w") as handle:
        for name, config, field, category, limit in sources:
            dataset = load_dataset(name, config, split="train", streaming=True)
            for item in collect(dataset, field, name, category, limit, seen):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Veda Pretraining Corpus v1 with provenance and exact deduplication")
    parser.add_argument("--output", type=Path, default=Path("data/pretraining-v1/documents.jsonl"))
    args = parser.parse_args()
    build(args.output)