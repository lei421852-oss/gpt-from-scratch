import argparse
from pathlib import Path

import torch

from gpt.data.tokenizer import CharacterTokenizer
from gpt.model.gpt import GPT
from gpt.utils.config import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with GPT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model.yaml",
    )
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    model = GPT(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = load_yaml(args.model_config).model
    corpus = (PROJECT_ROOT / "data" / "raw" / "input.txt").read_text(
        encoding="utf-8"
    )
    tokenizer = CharacterTokenizer(corpus)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            "Configured vocab_size does not match the tokenizer: "
            f"{model_config.vocab_size} != {tokenizer.vocab_size}"
        )

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except KeyError as error:
        raise ValueError(
            f"Prompt contains a character outside the vocabulary: {error.args[0]!r}"
        ) from error

    model = load_model(model_config, args.checkpoint, device)
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
