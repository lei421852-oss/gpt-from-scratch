import torch

from gpt.model.gpt import GPT


def test_gpt_forward():

    model = GPT(
        vocab_size=65,
        block_size=16,
        embedding_dim=256,
        num_layers=4,
        num_heads=8
    )


    x = torch.randint(
        0,
        65,
        (4,16)
    )


    logits = model(x)


    assert logits.shape == (
        4,
        16,
        65
    )