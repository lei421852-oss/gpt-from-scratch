# GPT From Scratch

A character-level, decoder-only GPT implementation built with PyTorch for
learning the model architecture and the engineering behind reproducible
training pipelines.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The training corpus is expected at `data/raw/input.txt`.

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
python scripts/train.py --resume experiments/<run>/checkpoints/last.pt
```

Model and training hyperparameters live in `configs/model.yaml` and
`configs/train.yaml`. Each run creates an ignored directory under
`experiments/` containing:

```text
<timestamp>-<name>/
├── configs/          # exact configuration snapshot
├── checkpoints/
│   ├── step_*.pt
│   └── last.pt
└── metrics.jsonl     # train and validation loss records
```

## Generate text

```bash
python scripts/generate.py \
  --checkpoint experiments/<run>/checkpoints/last.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 500 \
  --temperature 0.8 \
  --top-k 40
```

Use `--temperature 0` for deterministic greedy generation. Higher
temperatures increase randomness; `--top-k` restricts sampling to the most
likely tokens.

## Test

```bash
pytest
```
