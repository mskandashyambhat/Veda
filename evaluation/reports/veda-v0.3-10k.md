# Veda v0.3 10K Baseline

## Configuration

- Parameters: `26,548,608`
- Frozen tokenizer: 32K BPE
- Dataset: bounded BPE-tokenized TinyStories
- Training tokens: `1,216,162`
- Micro-batch: `2`
- Gradient accumulation: `16`
- Effective batch: `32`
- Context length: `256`
- Optimizer: AdamW
- Learning rate: `3e-4`
- Warmup: `500` steps
- Cosine-decay minimum: `3e-5`
- Gradient clipping: `1.0`
- Device: Apple M4 MPS
- Training time: approximately `15,598` seconds
- Tokens seen: `81,920,000`

## Learning Curve

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 500 | 3.4943 | 32.93 |
| 1000 | 2.7356 | 15.42 |
| 1500 | 2.6101 | 13.60 |
| 2000 | 2.6687 | 14.42 |
| 2500 | 3.0844 | 21.85 |
| 3000 | 3.0076 | 20.24 |
| 3500 | 3.4497 | 31.49 |
| 4000 | 3.4890 | 32.75 |
| 4500 | 3.7445 | 42.29 |
| 5000 | 4.0962 | 60.11 |
| 5500 | 4.1893 | 65.97 |
| 6000 | 4.2365 | 69.17 |
| 6500 | 4.5307 | 92.83 |
| 7000 | 4.4868 | 88.83 |
| 7500 | 4.6350 | 103.03 |
| 8000 | 4.6958 | 109.48 |
| 8500 | 4.7052 | 110.52 |
| 9000 | 4.8911 | 133.10 |
| 9500 | 4.8066 | 122.32 |
| 10000 | 4.5702 | 96.56 |

## Evaluation

The best checkpoint is `checkpoints/veda-v0.3-10k/best.pt`, selected at step `1500` with logged validation perplexity `13.60`. Independent evaluation over 20 random validation batches produced loss `2.6250` and perplexity `13.80`.

Fixed-prompt generations are stored in `evaluation/reports/step-best-v0.3-10k.json`.

The final step-10,000 training loss reached `0.1218`, but validation loss worsened after step 1,500. This run therefore demonstrates clear overfitting on the small bounded TinyStories training sample. The best checkpoint, not the final checkpoint, should be used for comparison and generation.

## Artifacts

- Checkpoints: `checkpoints/veda-v0.3-10k/`
- Metrics: `experiments/v0.3-10k/metrics.jsonl`
- Metadata: `checkpoints/veda-v0.3-10k/metadata.json`
- Independent evaluation: `evaluation/reports/step-best-v0.3-10k.json`