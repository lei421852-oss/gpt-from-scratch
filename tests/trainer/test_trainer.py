import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gpt.trainer.trainer import Trainer


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embedding(x)


def make_trainer_and_loader():
    model = TinyLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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
    optimizer_steps = 0
    original_step = trainer.optimizer.step

    def counted_step():
        nonlocal optimizer_steps
        optimizer_steps += 1
        original_step()

    trainer.optimizer.step = counted_step
    trainer.train(loader, max_steps=3, log_interval=10)
    assert optimizer_steps == 3


def test_evaluate_restores_training_mode():
    trainer, loader = make_trainer_and_loader()
    trainer.model.train()

    loss = trainer.evaluate(loader, max_batches=1)

    assert loss > 0
    assert trainer.model.training


def test_checkpoint_round_trip(tmp_path):
    trainer, loader = make_trainer_and_loader()
    trainer.train_step(*next(iter(loader)))
    expected_parameters = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, step=7)

    restored_trainer, _ = make_trainer_and_loader()
    step = restored_trainer.load_checkpoint(checkpoint_path)

    assert step == 7
    for expected, actual in zip(
        expected_parameters,
        restored_trainer.model.parameters(),
    ):
        assert torch.equal(expected, actual)
