from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from model.bpe_tokenizer import VedaBPETokenizer


def tiny_stories_texts(split: str, max_characters: int):
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    total = 0
    for row in dataset:
        text = row["text"].strip()
        if text:
            yield text
            total += len(text)
        if total >= max_characters:
            return


def corpus_texts(path: Path):
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            text = item["text"].strip()
            if text:
                yield text


def train(output: Path, vocab_size: int, corpus: Path | None, max_characters: int) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=VedaBPETokenizer.special_tokens,
        show_progress=True,
    )
    iterator = corpus_texts(corpus) if corpus else tiny_stories_texts("train", max_characters)
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    wrapper = VedaBPETokenizer(tokenizer)
    sample = "Hello, I am Veda."
    ids = wrapper.encode(sample)
    print(f"saved={output} vocab_size={tokenizer.get_vocab_size()} tokens={len(ids)} decoded={wrapper.decode(ids)!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Veda v0.2 BPE tokenizer")
    parser.add_argument("--output", type=Path, default=Path("artifacts/veda-v0.2-tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--max-characters", type=int, default=5_000_000)
    args = parser.parse_args()
    train(args.output, args.vocab_size, args.corpus, args.max_characters)