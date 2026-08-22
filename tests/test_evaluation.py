import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gpt.evaluation import evaluate_language_model


class FixedModel(nn.Module):
    def forward(self, x):
        logits = torch.zeros(*x.shape, 4)
        return logits


def test_evaluation_is_token_weighted_and_restores_mode():
    model = FixedModel()
    model.train()
    x = torch.tensor([[0, 1], [1, 2], [2, 3]])
    y = torch.tensor([[1, 2], [2, 3], [3, 0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_language_model(
        model,
        loader,
        torch.device("cpu"),
    )

    assert metrics["loss"] == pytest.approx(math.log(4))
    assert metrics["perplexity"] == pytest.approx(4)
    assert metrics["tokens"] == 6
    assert metrics["batches"] == 2
    assert model.training
