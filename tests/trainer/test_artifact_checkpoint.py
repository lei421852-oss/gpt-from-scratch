import torch
from torch import nn

from gpt.trainer.trainer import Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return x * self.weight


def make_trainer(metadata=None):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        model,
        optimizer,
        torch.device("cpu"),
        amp=False,
        artifact_metadata=metadata,
    )


def test_checkpoint_contains_portable_metadata(tmp_path):
    metadata = {
        "model_config": {"embedding_dim": 8},
        "tokenizer": {"format_version": 1},
    }
    trainer = make_trainer(metadata)
    path = tmp_path / "artifact.pt"
    trainer.save_checkpoint(path, step=3)

    checkpoint = torch.load(path, weights_only=False)

    assert checkpoint["artifact_format_version"] == 1
    assert checkpoint["model_config"] == metadata["model_config"]
    assert checkpoint["tokenizer"] == metadata["tokenizer"]


def test_checkpoint_restores_torch_random_state(tmp_path):
    torch.manual_seed(123)
    trainer = make_trainer()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, step=1)
    expected_random_value = torch.rand(1)

    torch.manual_seed(999)
    trainer.load_checkpoint(path)
    restored_random_value = torch.rand(1)

    assert torch.equal(restored_random_value, expected_random_value)
