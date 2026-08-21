import torch

from gpt.model.attention import CausalSelfAttention


def test_attention_shape():

    model = CausalSelfAttention(
        embedding_dim=256,
        block_size=16
    )


    x = torch.randn(
        4,
        16,
        256
    )


    y = model(x)


    assert y.shape == (
        4,
        16,
        256
    )