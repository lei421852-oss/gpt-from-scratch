from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gpt.data.dataset import LanguageModelDataset
from gpt.data.tokenizer import CharacterTokenizer
from gpt.model.gpt import GPT
from gpt.trainer.trainer import Trainer
from gpt.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    model_config = load_yaml(PROJECT_ROOT / "configs" / "model.yaml").model
    train_config = load_yaml(PROJECT_ROOT / "configs" / "train.yaml").training

    set_seed(train_config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    data_path = PROJECT_ROOT / "data" / "raw" / "input.txt"
    text = data_path.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer(text)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            "Configured vocab_size does not match the dataset: "
            f"{model_config.vocab_size} != {tokenizer.vocab_size}"
        )

    dataset = LanguageModelDataset(
        tokenizer.encode(text),
        block_size=model_config.block_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
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
    trainer = Trainer(model, optimizer, device)
    trainer.train(
        loader,
        max_steps=train_config.max_steps,
        log_interval=train_config.log_interval,
    )


if __name__ == "__main__":
    main()
