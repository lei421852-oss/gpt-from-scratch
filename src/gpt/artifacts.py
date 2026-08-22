from pathlib import Path

import torch

from gpt.data.tokenizer import CharacterTokenizer
from gpt.model.gpt import GPT


def load_model_artifact(path, device):
    checkpoint = torch.load(
        Path(path),
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
    return model, tokenizer, checkpoint
