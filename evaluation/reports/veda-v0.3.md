# Veda v0.3 Scaling Test

## Infrastructure

- Resumable model, optimizer, scheduler, step, best-loss, and RNG state.
- `latest.pt`, `best.pt`, and numbered step checkpoints.
- Micro-batch size: `2`.
- Gradient accumulation: `16`.
- Effective batch size: `32` sequences.
- Gradient clipping: maximum norm `1.0`.
- AdamW learning rate: `3e-4`.
- Warmup: `500` optimizer steps.
- Cosine decay minimum learning rate: `3e-5`.
- Weight decay: `0.1`.
- Automatic validation every 100 steps.
- NaN/Inf loss and exploding-gradient detection with emergency latest checkpoint.

## Model

- Parameters: `26,548,608`.
- Tokenizer: frozen 32K BPE tokenizer from v0.2.
- Architecture: 8 layers, 384 hidden size, 8 attention heads, 1,024 intermediate size.
- Dataset: bounded BPE-tokenized TinyStories.
- Training tokens: `1,216,162`.
- Device: Apple M4 MPS.
- Training time: approximately 11 minutes.

## Results

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 100 | 7.6195 | 2037.52 |
| 200 | 4.9949 | 147.66 |
| 300 | 4.2195 | 68.00 |
| 400 | 3.8659 | 47.74 |
| 500 | 3.4160 | 30.45 |

The 500-step scaling test completed without NaN/Inf losses or exploding gradients. Generation from `best.pt` also succeeded.

## Artifacts

- Checkpoints: `checkpoints/veda-v0.3/`
- Metrics: `experiments/v0.3/metrics.jsonl`
- Metadata: `checkpoints/veda-v0.3/metadata.json`

This is a scaling test, not a finished base-model training run. The next experiment should use the same infrastructure with a longer schedule and a mixed model-training corpus.