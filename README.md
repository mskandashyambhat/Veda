# Veda

Veda is a local language-model research project built from the ground up, with a future goal of becoming a local chat, coding, and editor system.

## Current Status

### Veda v0.1

- Decoder-only Transformer with RMSNorm, causal self-attention, SwiGLU, tied embeddings, and checkpoint loading.
- Byte-level tokenizer with a 258-token vocabulary.
- Approximately 4.95M parameters.
- Trained for 5,000 steps on a bounded TinyStories experiment.
- Best recorded validation perplexity: approximately 2.40.
- Apple M4 MPS training and evaluation.

### Veda v0.2

- Mixed-corpus byte-level BPE tokenizer with a 32,000-token vocabulary.
- Tokenizer corpus includes general text, technical writing, mathematics, and code-oriented data.
- BPE-aware dataset preparation, generation, and evaluation.
- Approximately 13.08M parameters.
- Completed 5,000-step MPS training run.
- Final independent validation perplexity: approximately 12.29.

### Veda v0.3 scaling test

- Approximately 26.55M parameters using the frozen 32K BPE tokenizer.
- Added resumable checkpoints, optimizer/scheduler/RNG restoration, gradient accumulation, gradient clipping, warmup/cosine decay, automatic validation, best-checkpoint selection, safety checks, and experiment metadata.
- Completed a 500-step MPS scaling test with effective batch size 32.
- Validation perplexity reached approximately 30.45 at step 500.

### Veda v0.3 10K baseline

- Completed 10,000 training steps with the same approximately 26.55M-parameter architecture.
- Best checkpoint occurred at step 1,500 with logged validation perplexity `13.60` and independently measured perplexity `13.80`.
- The final training loss reached `0.1218`, while validation worsened after step 1,500, clearly demonstrating overfitting on the bounded TinyStories sample.

Perplexity between v0.1 and v0.2 is not directly comparable because the tokenizers and vocabulary sizes differ.

## Project Structure

- `model/`: tokenizers and Transformer model implementation
- `training/`: corpus preparation, tokenizer training, and model training
- `evaluation/`: perplexity, generation, and tokenizer reports
- `configs/`: versioned model configurations
- `experiments/`: experiment notes and reproducibility information
- `scripts/`: generation commands

## Local Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch datasets tqdm pyyaml tokenizers
```

Run the model tests and tools with the project interpreter:

```bash
.venv/bin/python -m compileall -q model training evaluation scripts
```

Generate text from the v0.2 checkpoint after downloading or creating the local artifacts:

```bash
.venv/bin/python -m scripts.generate \
  --checkpoint checkpoints/veda-v0.2/step-005000/model.pt \
  --tokenizer artifacts/veda-v0.2-tokenizer-mixed.json \
  --prompt "Once upon a time" \
  --tokens 80
```

## Reports

- [Veda v0.1 report](evaluation/reports/veda-v0.1.md)
- [Veda v0.2 report](evaluation/reports/veda-v0.2.md)
- [Veda v0.3 scaling report](evaluation/reports/veda-v0.3.md)
- [Veda v0.3 10K report](evaluation/reports/veda-v0.3-10k.md)
- [Tokenizer report](evaluation/reports/tokenizer-v0.2.json)
- [Experiment notes](experiments/README.md)

## Roadmap

1. Improve training infrastructure with resume support, best-checkpoint selection, gradient accumulation, and learning-rate scheduling.
2. Scale the base model toward 25M and 50M parameters.
3. Add instruction tuning and local chat inference.
4. Build tool calling, codebase context, and the Veda editor.

## Collaborators

- **Skanda Shyam** - Creator and developer

## License

License information will be added before external distribution.
