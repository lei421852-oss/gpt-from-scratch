import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gpt.trainer.scheduler import WarmupCosineScheduler
from gpt.trainer.trainer import Trainer


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embedding(x)


def make_trainer_and_loader(with_scheduler=False):
    model = TinyLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = None
    if with_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer,
            max_learning_rate=0.1,
            min_learning_rate=0.01,
            warmup_steps=1,
            max_steps=4,
        )
    trainer = Trainer(
        model,
        optimizer,
        torch.device("cpu"),
        scheduler=scheduler,
        gradient_clip_norm=1.0,
        amp=True,
    )
    x = torch.tensor([[0, 1], [1, 2]])
    y = torch.tensor([[1, 2], [2, 3]])
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    return trainer, loader


def test_train_step_updates_parameters_and_records_gradient_norm():
    trainer, loader = make_trainer_and_loader()
    parameters_before = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]

    loss = trainer.train_step(*next(iter(loader)))

    assert loss > 0
    assert trainer.last_gradient_norm > 0
    assert not trainer.amp_enabled
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

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step
    trainer.train(loader, max_steps=3, log_interval=10)
    assert optimizer_steps == 3


def test_evaluate_restores_training_mode():
    trainer, loader = make_trainer_and_loader()
    trainer.model.train()

    loss = trainer.evaluate(loader, max_batches=1)

    assert loss > 0
    assert trainer.model.training


def test_checkpoint_restores_scheduler_and_optimizer(tmp_path):
    trainer, loader = make_trainer_and_loader(with_scheduler=True)
    trainer.scheduler.step(2)
    trainer.train_step(*next(iter(loader)))
    expected_parameters = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, step=2)

    restored_trainer, _ = make_trainer_and_loader(with_scheduler=True)
    step = restored_trainer.load_checkpoint(checkpoint_path)

    assert step == 2
    assert restored_trainer.scheduler.step_number == 2
    assert restored_trainer.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]
    for expected, actual in zip(
        expected_parameters,
        restored_trainer.model.parameters(),
    ):
        assert torch.equal(expected, actual)
