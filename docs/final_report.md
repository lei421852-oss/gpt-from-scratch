# GPT From Scratch — Final Experiment Report

## Baseline

- Dataset: Tiny Shakespeare
- Tokenizer: character-level, 65 tokens
- Context length: 128
- Embedding dimension: 256
- Layers: 6
- Attention heads: 8
- Parameters: approximately 4.8 million
- Training steps: 10,000
- Batch size: 32
- Effective predictions per step: 4,096 tokens
- Precision: BF16 on CUDA
- Optimizer: AdamW with decoupled parameter groups
- Schedule: 500-step linear warmup followed by cosine decay
- Gradient clipping: global norm 1.0

## Results

Training completed without NaN values, gradient explosion, or interruption.

| Metric | Result |
|---|---:|
| Final train loss | 1.2223 |
| Full validation loss (`last.pt`) | 1.4849 |
| Full validation perplexity (`last.pt`) | 4.4147 |
| Full validation loss (`best.pt`) | 1.4864 |
| Full validation perplexity (`best.pt`) | 4.4211 |
| Validation tokens evaluated | 14,260,736 |

The final checkpoint slightly outperformed the checkpoint selected from the
original fast-monitoring subset. Investigation showed that monitoring only
the first validation batches was optimistic. The final code now samples a
fixed set of windows uniformly across the complete validation region.

## Interpretation

The validation curve improved throughout training and approached a plateau
near the end. Train and validation losses remained reasonably close, so the
run shows no strong overfitting signal. The final learning rate reached the
configured minimum of `3e-5`, and pre-clipping gradient norms remained stable.

Generated text demonstrates that the model learned:

- speaker names and colon-delimited dialogue;
- line breaks and play-like formatting;
- English-like words and local grammar;
- recurring Shakespeare-style vocabulary.

Greedy generation is coherent but repetitive. Temperature `0.8` with
`top-k=40` provides a better balance between structure and diversity.

## Recommended artifact

Use:

```text
experiments/20260822-090751-719841-final-baseline/checkpoints/last.pt
```

Representative samples are stored under the same run's `samples/` directory.

## Limitations

- Character tokenization is inefficient compared with BPE.
- Context is limited to 128 characters.
- The dataset is small and contains overlapping training windows.
- Fast validation is an estimate; final decisions should use full evaluation.
- Ordinary random sampling cannot resume mid-epoch with strict bitwise identity.
- This model learns one small corpus and is not a general-purpose language model.

## Recommended future work

The most educational next extension is a GPT-2-compatible project with BPE,
weight tying, PyTorch scaled dot-product attention, and pretrained-weight
loading. Larger-scale work can then add gradient accumulation, compilation,
distributed training, and Hugging Face export.
