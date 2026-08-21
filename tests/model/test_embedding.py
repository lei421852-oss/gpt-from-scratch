import torch

from gpt.model.embedding import GPTEmbedding


def test_embedding_shape():

    model = GPTEmbedding(
        vocab_size=65,
        embedding_dim=256,
        block_size=16
    )


    x = torch.randint(
        0,
        65,
        (4,16)
    )


    output = model(x)


    assert output.shape == (
        4,
        16,
        256
    )