import argparse
from pathlib import Path

import torch

from gpt.data.tokenizer import CharacterTokenizer
from gpt.model.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with GPT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_artifact(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    required_fields = {
        "artifact_format_version",
        "model_config",
        "tokenizer",
        "model_state_dict",
    }
    missing_fields = required_fields - checkpoint.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Checkpoint is not a portable model artifact; missing: {missing}"
        )
    if checkpoint["artifact_format_version"] != 1:
        raise ValueError("Unsupported model artifact format version")

    config = checkpoint["model_config"]
    tokenizer = CharacterTokenizer.from_dict(checkpoint["tokenizer"])
    if tokenizer.vocab_size != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match model config")

    model = GPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_artifact(args.checkpoint, device)

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
