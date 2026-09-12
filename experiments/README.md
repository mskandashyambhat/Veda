# Veda Experiments

Each experiment records its configuration, metrics, checkpoint steps, and evaluation samples.

## v0.1

- Byte tokenizer, vocabulary 258
- TinyStories bounded corpus
- 4,951,808 parameters
- 5,000 training steps
- Report: `evaluation/reports/veda-v0.1.md`

## v0.2

- Mixed-corpus byte-level BPE tokenizer, vocabulary 32,000
- Tokenizer corpus: 15,000,000 characters from TinyStories, FineWeb-Edu, OpenWebMath, and SmolTalk
- TinyStories model-training sample: 1,216,162 BPE training tokens and 122,143 validation tokens
- 13,077,760 parameters
- Metrics: `experiments/v0.2/metrics.jsonl`
- Checkpoints: `checkpoints/veda-v0.2/step-*`

Perplexity should not be compared as an absolute apples-to-apples score between v0.1 and v0.2 because their tokenizers and vocabulary sizes differ. Compare it alongside token counts, throughput, memory, and fixed-prompt generations.