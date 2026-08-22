# GPT From Scratch

A character-level, decoder-only GPT implemented with PyTorch. The project
covers the complete path from raw text and causal self-attention to
reproducible training, portable checkpoints, evaluation, and generation.

## Features

- Character tokenizer with portable JSON serialization
- Decoder-only GPT with learned token and position embeddings
- Multi-head causal self-attention and pre-norm residual blocks
- Configuration-driven train/validation pipeline
- AdamW parameter grouping, warmup, and cosine learning-rate decay
- Gradient clipping and automatic BF16/FP16 mixed precision
- Experiment metrics, configuration snapshots, and best checkpoints
- Resumable training with optimizer, scheduler, scaler, and RNG states
- Token-weighted loss and perplexity evaluation
- Greedy, temperature, and top-k autoregressive generation

## Project structure

```text
configs/                 Model and training configuration
data/raw/input.txt       Tiny Shakespeare corpus (not committed)
scripts/
  download_data.py       Download the corpus
  validate_data.py       Validate the corpus
  train.py               Train or resume a model
  evaluate.py            Evaluate a portable checkpoint
  generate.py            Generate text
src/gpt/
  data/                   Tokenizer, dataset, and loader
  model/                  GPT architecture
  trainer/                Optimizer, scheduler, and trainer
  artifacts.py            Portable checkpoint loading
  evaluation.py           Token-weighted evaluation
  utils/                  Configuration and experiment management
tests/                    Automated test suite
```

## Environment

- Python 3.10+
- PyTorch 2.13
- CUDA-capable GPU recommended; CPU execution is supported

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Prepare data

```bash
python scripts/download_data.py
python scripts/validate_data.py
```

The expected corpus path is `data/raw/input.txt`.

## Configuration

`configs/model.yaml` controls model architecture. `configs/train.yaml`
controls optimization, evaluation, checkpoints, and experiment naming.

Important training settings include:

- `learning_rate`, `min_learning_rate`, and `warmup_steps`
- `weight_decay`, `beta1`, and `beta2`
- `gradient_clip_norm` and `amp`
- `max_steps`, `eval_interval`, and `checkpoint_interval`

## Train

```bash
python scripts/train.py
```

Override the experiment name:

```bash
python scripts/train.py --experiment-name wider-model
```

Resume from a checkpoint:

```bash
python scripts/train.py \
  --resume experiments/<run>/checkpoints/last.pt
```

Each run creates an ignored directory:

```text
experiments/<timestamp>-<name>/
├── configs/              Exact configuration snapshot
├── checkpoints/
│   ├── best.pt           Lowest monitored validation loss
│   ├── step_*.pt         Periodic checkpoints
│   └── last.pt           Final training state
├── metrics.jsonl         Train and validation metrics
├── summary.json          Best and final results
└── tokenizer.json        Human-readable tokenizer vocabulary
```

## Evaluate

The evaluation command processes the complete validation split and reports
token-weighted loss and perplexity:

```bash
python scripts/evaluate.py \
  --checkpoint experiments/<run>/checkpoints/best.pt
```

## Generate text

Portable checkpoints contain model configuration and tokenizer state, so
generation does not require the training corpus or YAML files.

```bash
python scripts/generate.py \
  --checkpoint experiments/<run>/checkpoints/best.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 500 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42
```

Use `--temperature 0` for deterministic greedy generation. Higher
temperatures increase diversity; `--top-k` excludes low-probability tokens.

## Baseline result

The completed 10,000-step baseline uses 6 layers, 256 embedding dimensions,
8 attention heads, a 128-token context, and approximately 4.8 million
parameters.

- Final train loss: `1.2223`
- Full validation loss: `1.4849`
- Full validation perplexity: `4.4147`
- Evaluated validation tokens: `14,260,736`

The model generates recognizable Shakespeare-style speaker labels, dialogue,
punctuation, and line structure. The final checkpoint slightly outperformed
the monitored best checkpoint on the complete validation set and is the
recommended artifact for this baseline.

## Quality checks

```bash
pytest
ruff check .
```

## Scope and limitations

This is a small educational GPT trained on Tiny Shakespeare. It is designed
to make the architecture and engineering workflow understandable, not to
compete with pretrained production language models. The tokenizer is
character-level, the context length is short, and training uses one GPU.
