from __future__ import annotations

import argparse
import json
import math
import random
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


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["mps"].cpu())


def load_config(path: Path | None, args: argparse.Namespace) -> VedaConfig:
    if path:
        values = yaml.safe_load(path.read_text())
        fields = VedaConfig.__dataclass_fields__
        return VedaConfig(**{key: values[key] for key in fields if key in values})
    return VedaConfig(context_length=args.context_length, vocab_size=args.vocab_size, hidden_size=args.hidden_size, num_layers=args.num_layers, intermediate_size=args.intermediate_size)


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, minimum_lr: float):
    initial_lr = optimizer.param_groups[0]["lr"]

    def schedule(step: int) -> float:
        if step <= warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-8)
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (minimum_lr + (initial_lr - minimum_lr) * cosine) / initial_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def validate(model: VedaModel, tokens: np.ndarray, batch_size: int, batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = get_batch(tokens, batch_size, model.config.context_length, device)
            _, loss = model(inputs, targets)
            losses.append(float(loss.detach()))
    model.train()
    return sum(losses) / len(losses)


def checkpoint_payload(model, optimizer, scheduler, config, step, best_validation_loss, metrics, device):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "best_validation_loss": best_validation_loss,
        "metrics": metrics,
        "rng_state": capture_rng_state(),
        "device": str(device),
    }


def save_checkpoint(root: Path, name: str, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    torch.save(payload, root / name)


def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    train_tokens = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    validation_tokens = np.memmap(data_dir / "validation.bin", dtype=np.uint16, mode="r")
    config = load_config(args.config, args)
    model = VedaModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.steps, args.minimum_lr)
    checkpoint_root = Path(args.checkpoint_dir)
    resume_path = Path(args.resume) if args.resume else None
    start_step = 0
    best_validation_loss = float("inf")
    history: list[dict] = []
    if resume_path:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_step = int(saved["step"])
        best_validation_loss = float(saved["best_validation_loss"])
        history = saved.get("metrics", [])
        restore_rng_state(saved["rng_state"])
        print(f"resumed={resume_path} step={start_step} best_validation_loss={best_validation_loss:.4f}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "model": args.model_name,
        "parameters": parameters,
        "tokenizer": args.tokenizer,
        "dataset": args.dataset,
        "training_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "steps": args.steps,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "context_length": config.context_length,
        "learning_rate": args.learning_rate,
        "minimum_lr": args.minimum_lr,
        "warmup_steps": args.warmup_steps,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "hardware": str(device),
    }
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics_file = Path(args.metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    if not resume_path:
        metrics_file.write_text("")
    model.train()
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            inputs, targets = get_batch(train_tokens, args.batch_size, config.context_length, device)
            _, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                payload = checkpoint_payload(model, optimizer, scheduler, config, step - 1, best_validation_loss, history, device)
                save_checkpoint(checkpoint_root, "latest.pt", payload)
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(gradient_norm) or gradient_norm > args.max_gradient_norm:
            payload = checkpoint_payload(model, optimizer, scheduler, config, step - 1, best_validation_loss, history, device)
            save_checkpoint(checkpoint_root, "latest.pt", payload)
            raise FloatingPointError(f"Exploding gradient at step {step}: {gradient_norm}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss = accumulated_loss / args.gradient_accumulation_steps
        elapsed = time.perf_counter() - start_time
        record = {"step": step, "train_loss": train_loss, "gradient_norm": gradient_norm, "learning_rate": scheduler.get_last_lr()[0], "tokens_per_sec": step * args.batch_size * args.gradient_accumulation_steps * config.context_length / elapsed}
        if step % args.validation_every == 0 or step == args.steps:
            validation_loss = validate(model, validation_tokens, args.validation_batch_size, args.validation_batches, device)
            record.update({"validation_loss": validation_loss, "perplexity": math.exp(validation_loss), "training_tokens_seen": step * args.batch_size * args.gradient_accumulation_steps * config.context_length})
            history.append(record)
            payload = checkpoint_payload(model, optimizer, scheduler, config, step, min(best_validation_loss, validation_loss), history, device)
            save_checkpoint(checkpoint_root, "latest.pt", payload)
            save_checkpoint(checkpoint_root, f"step-{step:06d}.pt", payload)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                payload["best_validation_loss"] = best_validation_loss
                save_checkpoint(checkpoint_root, "best.pt", payload)
            print(f"step={step} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} perplexity={math.exp(validation_loss):.2f} lr={scheduler.get_last_lr()[0]:.2e} grad_norm={gradient_norm:.2f}")
        elif step == 1 or step % args.log_every == 0:
            history.append(record)
            print(f"step={step} train_loss={train_loss:.4f} tokens_per_sec={record['tokens_per_sec']:.0f} lr={scheduler.get_last_lr()[0]:.2e} grad_norm={gradient_norm:.2f}")
        if step % args.log_every == 0 or step == args.steps:
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
    metadata["training_seconds"] = time.perf_counter() - start_time
    metadata["best_validation_loss"] = best_validation_loss
    (checkpoint_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Veda with resumable checkpoints and validation")
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--checkpoint-dir", default="checkpoints/veda-v0.3")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=258)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=704)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model-name", default="Veda")
    parser.add_argument("--tokenizer", default="byte")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=100.0)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--metrics-file", default="experiments/v0.3/metrics.jsonl")
    train(parser.parse_args())
