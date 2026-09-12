# Veda v0.2 BPE Model

## Tokenizer

- Tokenizer: mixed-corpus byte-level BPE
- Vocabulary: `32,000`
- Artifact: `artifacts/veda-v0.2-tokenizer-mixed.json`
- Corpus: 15,000,000 characters across TinyStories, FineWeb-Edu, OpenWebMath, and SmolTalk
- All required English, code, JSON, SQL, Markdown, mathematics, Hindi, and Japanese samples round-trip correctly.

The BPE corpus contains 15,000,000 characters and compresses to 3,507,811 BPE tokens, compared with 15,068,090 byte tokens in the per-source report. The exact source-level counts are stored in `evaluation/reports/tokenizer-v0.2.json`.

## Model Run

- Architecture kept close to v0.1
- Parameters: `13,077,760`
- Context length: 256
- Dataset: the same bounded TinyStories sample, re-tokenized with BPE
- Training tokens: `1,216,162`
- Validation tokens: `122,143`
- Device: Apple M4 MPS
- Steps: `5,000`
- Checkpoints: `checkpoints/veda-v0.2/step-*`
- Metrics: `experiments/v0.2/metrics.jsonl`

## Learning Curve

| Step | Validation loss | Logged perplexity |
| ---: | ---: | ---: |
| 500 | 3.7322 | 41.77 |
| 1000 | 3.2246 | 25.14 |
| 1500 | 3.0758 | 21.67 |
| 2000 | 2.7283 | 15.31 |
| 2500 | 2.6588 | 14.28 |
| 3000 | 2.5678 | 13.04 |
| 3500 | 2.6583 | 14.27 |
| 4000 | 2.5109 | 12.32 |
| 4500 | 2.5506 | 12.81 |
| 5000 | 2.6191 | 13.72 |

Independent evaluation of the final checkpoint over 20 random validation batches produced loss `2.5087` and perplexity `12.29`. Fixed-prompt generations are in `evaluation/reports/step-5000-v0.2.json`.

## v0.1 Comparison

| Measurement | v0.1 | v0.2 |
| --- | ---: | ---: |
| Tokenizer | Byte | 32K BPE |
| Parameters | 4,951,808 | 13,077,760 |
| Model training tokens | 5,023,968 | 1,216,162 |
| Final independent perplexity | 2.40 | 12.29 |
| Approx. final throughput | 17.3k tokens/sec | 9.1k tokens/sec |

Perplexity is not directly comparable across these runs because the token definitions and vocabulary sizes changed. v0.2 uses substantially fewer tokens for the same text and produces longer, more coherent fixed-prompt continuations, but it also has a larger output vocabulary and was trained on fewer token updates.

## Status

Veda v0.2 validates the 32K BPE tokenizer, BPE data preparation, expanded model vocabulary, MPS training, checkpoint loading, and BPE-aware generation. The next training-infrastructure improvements should be resume support, best-checkpoint selection, gradient accumulation, and a warmup/cosine scheduler before scaling the model.