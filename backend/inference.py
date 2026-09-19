from __future__ import annotations

from pathlib import Path

import torch

from model.bpe_tokenizer import VedaBPETokenizer
from model.config import VedaConfig
from model.veda import VedaModel


class VedaProvider:
    def __init__(self, checkpoint: str | Path | None = None, tokenizer: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint or "checkpoints/veda-v0.3-10k/best.pt")
        self.tokenizer_path = Path(tokenizer or "artifacts/veda-v0.2-tokenizer-mixed.json")
        self.model: VedaModel | None = None
        self.tokenizer: VedaBPETokenizer | None = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint_path.exists() or not self.tokenizer_path.exists():
            return
        saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model = VedaModel(VedaConfig(**saved["config"])).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.tokenizer = VedaBPETokenizer.from_file(self.tokenizer_path)

    @property
    def ready(self) -> bool:
        self.load()
        return self.model is not None and self.tokenizer is not None

    def generate(self, prompt: str, max_tokens: int = 120, temperature: float = 0.8) -> str:
        self.load()
        if not self.ready:
            return "Veda local checkpoint is unavailable. Place a trained checkpoint at checkpoints/veda-v0.3-10k/best.pt."
        assert self.model is not None and self.tokenizer is not None
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            for _ in range(max_tokens):
                context = input_ids[:, -self.model.config.context_length :]
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / max(temperature, 0.05)
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                if next_token.item() == self.tokenizer.eos_id:
                    break
        return self.tokenizer.decode(input_ids[0].tolist())
