from dataclasses import dataclass


@dataclass
class VedaConfig:
    vocab_size: int = 258
    context_length: int = 256
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 704
    dropout: float = 0.0