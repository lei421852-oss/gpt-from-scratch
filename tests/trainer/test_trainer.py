import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gpt.trainer.trainer import Trainer


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embedding(x)


def make_trainer_and_loader():
    model = TinyLanguageModel(vocab_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, torch.device("cpu"))

    x = torch.tensor([[0, 1], [1, 2]])
    y = torch.tensor([[1, 2], [2, 3]])
    loader = DataLoader(TensorDataset(x, y), batch_size=1)

    return trainer, loader


def test_train_step_updates_model_parameters():
    trainer, loader = make_trainer_and_loader()
    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]

    loss = trainer.train_step(*next(iter(loader)))

    assert loss > 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            trainer.model.parameters(),
        )
    )


def test_train_stops_at_max_steps():
    trainer, loader = make_trainer_and_loader()
