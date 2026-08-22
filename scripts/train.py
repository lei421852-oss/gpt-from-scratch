import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gpt.data.dataset import LanguageModelDataset
from gpt.data.tokenizer import CharacterTokenizer
from gpt.model.gpt import GPT
from gpt.trainer.trainer import Trainer
from gpt.utils.config import load_yaml
from gpt.utils.experiment import Experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT from scratch")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model.yaml",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override training.experiment_name",
    )
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_tokens(tokens, validation_split, block_size):
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between zero and one")

    split_index = int(len(tokens) * (1 - validation_split))
    train_tokens = tokens[:split_index]
    validation_tokens = tokens[split_index:]

    minimum_length = block_size + 1
    if len(train_tokens) < minimum_length or len(validation_tokens) < minimum_length:
        raise ValueError("Both dataset splits must be longer than block_size")

    return train_tokens, validation_tokens


def main():
    args = parse_args()
    model_config = load_yaml(args.model_config).model
    train_config = load_yaml(args.train_config).training

    set_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = (PROJECT_ROOT / "data" / "raw" / "input.txt").read_text(
        encoding="utf-8"
    )
    tokenizer = CharacterTokenizer(text)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            "Configured vocab_size does not match the dataset: "
            f"{model_config.vocab_size} != {tokenizer.vocab_size}"
        )

    train_tokens, validation_tokens = split_tokens(
        tokenizer.encode(text),
        train_config.validation_split,
        model_config.block_size,
    )
    train_dataset = LanguageModelDataset(
        train_tokens,
        block_size=model_config.block_size,
    )
    validation_dataset = LanguageModelDataset(
        validation_tokens,
        block_size=model_config.block_size,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    model = GPT(
        vocab_size=model_config.vocab_size,
        block_size=model_config.block_size,
        embedding_dim=model_config.embedding_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
    )

    experiment_name = args.experiment_name or train_config.experiment_name
    experiment = Experiment(
        PROJECT_ROOT / "experiments",
        experiment_name,
        config_paths=(args.model_config, args.train_config),
    )
    print(f"Experiment: {experiment.directory}")

    trainer = Trainer(model, optimizer, device, experiment=experiment)
    start_step = trainer.load_checkpoint(args.resume) if args.resume else 0
    if args.resume:
        print(f"Resumed from step {start_step}: {args.resume}")

    trainer.train(
        train_loader,
        max_steps=train_config.max_steps,
        log_interval=train_config.log_interval,
        validation_loader=validation_loader,
        eval_interval=train_config.eval_interval,
        eval_batches=train_config.eval_batches,
        checkpoint_interval=train_config.checkpoint_interval,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()
