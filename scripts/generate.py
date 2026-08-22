import argparse
from pathlib import Path

import torch

from gpt.artifacts import load_model_artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with GPT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model_artifact(args.checkpoint, device)

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except KeyError as error:
        raise ValueError(
            f"Prompt contains a character outside the vocabulary: {error.args[0]!r}"
        ) from error

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
