import torch

from gpt.model.multi_head_attention import MultiHeadCausalSelfAttention


def test_multi_head_attention_shape():

    model = MultiHeadCausalSelfAttention(
        embedding_dim=256,
        num_heads=8,
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