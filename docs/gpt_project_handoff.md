# GPT From Scratch — Project Handoff

## Status

The main project is complete. It includes data preparation, tokenizer,
decoder-only GPT architecture, optimized training, experiment management,
portable checkpoints, evaluation, and text generation.

## Environment

- WSL Ubuntu 24.04
- Python 3.12
- PyTorch 2.13 with CUDA 13.0
- CUDA BF16 training verified
- NVIDIA GeForce RTX 5050 Laptop GPU

## Completed stages

1. Repository and package initialization
2. Tiny Shakespeare data pipeline
3. Character tokenizer
4. Next-token language-model dataset
5. DataLoader pipeline
6. GPT architecture from scratch
7. Basic training loop
8. Configuration-driven training and experiment management
9. Autoregressive text generation
10. AdamW grouping, warmup/cosine scheduling, clipping, and AMP
11. Portable model artifacts and RNG state restoration
12. Full evaluation, best checkpoints, baseline training, and analysis
13. Code cleanup, documentation, and final verification

## Final baseline

- Run: `20260822-090751-719841-final-baseline`
- Model parameters: approximately 4.8 million
- Final step: 10,000
- Final train loss: 1.2223
- Full validation loss: 1.4849
- Full validation perplexity: 4.4147
- Recommended checkpoint: `last.pt`

## Verification

- Full automated test suite passes
- Full Ruff code-quality check passes
- Git diff whitespace check passes
- CUDA BF16 training passes
- Checkpoint resume passes
- Complete validation evaluation passes
- Portable generation without YAML or corpus passes

## Commands

```bash
# Train
python scripts/train.py

# Resume
python scripts/train.py --resume experiments/<run>/checkpoints/last.pt

# Evaluate
python scripts/evaluate.py --checkpoint experiments/<run>/checkpoints/last.pt

# Generate
python scripts/generate.py \
  --checkpoint experiments/<run>/checkpoints/last.pt \
  --prompt "ROMEO:" \
  --temperature 0.8 \
  --top-k 40

# Verify
pytest
ruff check .
```

## Known boundary

The project is an educational character-level GPT, not a general-purpose LLM.
The next recommended project is a GPT-2-compatible implementation using BPE
and pretrained weight loading.
