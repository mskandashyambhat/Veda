from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from model.config import VedaConfig
from model.veda import VedaModel


def get_batch(tokens: np.ndarray, batch_size: int, context_length: int, device: torch.device):
    starts = np.random.randint(0, len(tokens) - context_length - 1, size=batch_size)
    inputs = np.stack([tokens[start : start + context_length] for start in starts])
    targets = np.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return torch.from_numpy(inputs.astype(np.int64)).to(device), torch.from_numpy(targets.astype(np.int64)).to(device)


def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    train_tokens = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    validation_tokens = np.memmap(data_dir / "validation.bin", dtype=np.uint16, mode="r")
    if args.config:
        config_values = yaml.safe_load(Path(args.config).read_text())
        config = VedaConfig(**{key: config_values[key] for key in VedaConfig.__dataclass_fields__ if key in config_values})
    else:
        config = VedaConfig(context_length=args.context_length, vocab_size=args.vocab_size)
    model = VedaModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters={parameters:,} train_tokens={len(train_tokens):,}")
    metrics_file = Path(args.metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text("")
    model.train()
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(train_tokens, args.batch_size, config.context_length, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = step * args.batch_size * config.context_length / elapsed
            record = {"step": step, "train_loss": float(loss.detach()), "tokens_per_sec": tokens_per_sec}
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"step={step} loss={loss.item():.4f} tokens_per_sec={tokens_per_sec:.0f}")
        if step % args.save_every == 0 or step == args.steps:
            checkpoint_dir = Path(args.checkpoint_dir)
            step_dir = checkpoint_dir / f"step-{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config.__dict__, "step": step}, step_dir / "model.pt")
            torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
            with torch.no_grad():
                validation_inputs, validation_targets = get_batch(validation_tokens, args.batch_size, config.context_length, device)
                _, validation_loss = model(validation_inputs, validation_targets)
            validation_record = {"step": step, "train_loss": float(loss.detach()), "validation_loss": float(validation_loss), "perplexity": math.exp(float(validation_loss)), "training_tokens": step * args.batch_size * config.context_length, "device": str(device)}
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(validation_record) + "\n")
            (step_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2) + "\n")
            (step_dir / "metadata.json").write_text(json.dumps(validation_record, indent=2) + "\n")
            print(f"checkpoint={step_dir / 'model.pt'} validation_loss={validation_loss.item():.4f} perplexity={math.exp(validation_loss.item()):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the first Veda language model")
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--checkpoint-dir", default="checkpoints/veda-10m")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=258)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--metrics-file", default="training/metrics.jsonl")
    train(parser.parse_args())