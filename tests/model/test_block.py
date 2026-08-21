import torch

from gpt.model.block import GPTBlock


def test_gpt_block_shape():

    block = GPTBlock(
        embedding_dim=256,
        num_heads=8,
        block_size=16
    )


    x = torch.randn(
        4,
        16,
        256
    )


    y = block(x)


    assert y.shape == (
        4,
        16,
        256
    )