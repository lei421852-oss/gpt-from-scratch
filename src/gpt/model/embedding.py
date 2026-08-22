import torch
from torch import nn


class GPTEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        block_size
    ):
        super().__init__()


        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )


        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )


    def forward(self, x):

        _batch_size, seq_len = x.shape


        token_emb = self.token_embedding(x)


        positions = torch.arange(
            seq_len,
            device=x.device
        )


        pos_emb = self.position_embedding(
            positions
        )


        return token_emb + pos_emb