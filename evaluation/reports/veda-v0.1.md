# Veda v0.1 Base Model

## Run

- Dataset: `roneneldan/TinyStories`, bounded subset
- Training tokens: 5,023,968
- Validation tokens: 502,115
- Tokenizer: byte-level, vocabulary size 258
- Parameters: 4,951,808
- Device: Apple M4 MPS
- Architecture: 6-layer decoder-only Transformer, 256 hidden size, 8 attention heads, RMSNorm, SwiGLU, learned position embeddings
- Training command: `.venv/bin/python -m training.train --steps 5000 --save-every 500 --log-every 50`

## Learning Curve

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 100 | 2.3671 | 10.67 |
| 500 | 1.9523 | 7.04 |
| 1000 | 1.2664 | 3.55 |
| 1500 | 1.1731 | 3.23 |
| 2000 | 1.0428 | 2.84 |
| 2500 | 0.9739 | 2.65 |
| 3000 | 0.9031 | 2.47 |
| 3500 | 0.9743 | 2.65 |
| 4000 | 0.9045 | 2.47 |
| 4500 | 0.8738 | 2.40 |
| 5000 | 0.9229 | 2.52 |

The final checkpoint was independently evaluated with 20 random validation batches: loss `0.8772`, perplexity `2.4041`.

## Artifact

The final weights are at `checkpoints/veda-10m/veda.pt`. The directory name is historical; the actual model has 4.95M parameters.

Fixed-prompt generations are stored in `evaluation/reports/step-5000.json`.

The original 5,000-step run overwrote the same checkpoint at each save, so intermediate weights for generation comparison were not retained. Future runs use the reproducible step-directory layout implemented in `training/train.py`.

## Status

Veda v0.1 proves that the tokenizer, Transformer, MPS training loop, validation, checkpoint loading, and generation pipeline work. It is a TinyStories base-model experiment, not yet an instruction-tuned assistant.