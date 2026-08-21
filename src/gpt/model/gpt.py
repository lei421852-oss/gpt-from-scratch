import torch
import torch.nn as nn

from gpt.model.embedding import GPTEmbedding
from gpt.model.block import GPTBlock


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim,
        num_layers,
        num_heads,
        dropout=0.1
    ):
        super().__init__()


        self.embedding = GPTEmbedding(
            vocab_size,
            embedding_dim,
            block_size
        )


        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    embedding_dim,
                    num_heads,
                    block_size,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )


        self.ln_f = nn.LayerNorm(
            embedding_dim
        )


        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )


    def forward(self, x):

        x = self.embedding(x)


        for block in self.blocks:
            x = block(x)


        x = self.ln_f(x)


        logits = self.lm_head(x)


        return logits