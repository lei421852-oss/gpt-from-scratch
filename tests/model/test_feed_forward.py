import torch

from gpt.model.feed_forward import FeedForward


def test_feed_forward_shape():

    model = FeedForward(
        embedding_dim=256
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