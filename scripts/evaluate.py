import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gpt.artifacts import load_model_artifact
from gpt.data.dataset import LanguageModelDataset
from gpt.evaluation import evaluate_language_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a GPT artifact")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "input.txt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, checkpoint = load_model_artifact(args.checkpoint, device)
    training_config = checkpoint.get("training_config")
    if training_config is None:
        raise ValueError("Artifact does not contain training_config")

    text = args.data.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    split_index = int(
        len(tokens) * (1 - training_config["validation_split"])
    )
    validation_tokens = tokens[split_index:]
    dataset = LanguageModelDataset(
        validation_tokens,
        block_size=checkpoint["model_config"]["block_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    metrics = evaluate_language_model(
        model,
        loader,
        device,
        amp=training_config.get("amp", False),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
