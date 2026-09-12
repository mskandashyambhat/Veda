from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.benchmarks import run_generation_benchmark
from evaluation.perplexity import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Veda checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", default="data/tinystories/validation.bin")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    report = {"checkpoint": args.checkpoint, "perplexity": evaluate(args.checkpoint, args.data_file), "generation": run_generation_benchmark(args.checkpoint, tokenizer_path=args.tokenizer)}
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()