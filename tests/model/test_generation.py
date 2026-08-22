import pytest
import torch

from gpt.model.gpt import GPT


def make_model():
    return GPT(
        vocab_size=5,
        block_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )


def test_generate_appends_requested_number_of_tokens():
    model = make_model()
    input_ids = torch.tensor([[0, 1, 2]])

    generated = model.generate(
        input_ids,
        max_new_tokens=6,
        temperature=0,
    )

    assert generated.shape == (1, 9)
    assert torch.equal(generated[:, :3], input_ids)


def test_generate_restores_training_mode():
    model = make_model()
    model.train()

    model.generate(
        torch.tensor([[0]]),
        max_new_tokens=1,
        top_k=2,
    )

    assert model.training


def test_greedy_generation_is_deterministic():
    model = make_model()
    input_ids = torch.tensor([[0, 1]])

    first = model.generate(input_ids, 4, temperature=0)
    second = model.generate(input_ids, 4, temperature=0)

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": 1, "temperature": -0.1}, "temperature"),
        ({"max_new_tokens": 1, "top_k": 0}, "top_k"),
    ],
)
def test_generate_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_model().generate(torch.tensor([[0]]), **kwargs)
