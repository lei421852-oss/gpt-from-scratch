import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim


        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3
        )


        self.proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )


        self.dropout = nn.Dropout(dropout)


        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )


    def forward(self, x):

        B,T,C = x.shape


        qkv = self.qkv(x)


        q,k,v = torch.chunk(
            qkv,
            3,
            dim=-1
        )


        attention_scores = (
            q @ k.transpose(-2,-1)
        ) / (C ** 0.5)


        attention_scores = attention_scores.masked_fill(
            self.mask[:T,:T] == 0,
            float("-inf")
        )


        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )


        attention_weights = self.dropout(
            attention_weights
        )


        output = attention_weights @ v


        return self.proj(output)