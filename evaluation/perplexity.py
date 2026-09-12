from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from model.config import VedaConfig
from model.veda import VedaModel


def load_model(checkpoint: str | Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = VedaModel(VedaConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def evaluate(checkpoint: str | Path, data_file: str | Path, batch_size: int = 8, batches: int = 20) -> dict[str, float]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(checkpoint, device)
    tokens = np.memmap(data_file, dtype=np.uint16, mode="r")
    context_length = model.config.context_length
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(batches):
            starts = np.random.randint(0, len(tokens) - context_length - 1, size=batch_size)
            inputs = torch.from_numpy(np.stack([tokens[start : start + context_length] for start in starts]).astype(np.int64)).to(device)
            targets = torch.from_numpy(np.stack([tokens[start + 1 : start + context_length + 1] for start in starts]).astype(np.int64)).to(device)
            _, loss = model(inputs, targets)
            losses.append(float(loss))
    loss = sum(losses) / len(losses)
    return {"loss": loss, "perplexity": math.exp(loss), "batches": batches}