import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0


        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads


        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
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


    def forward(self,x):

        B,T,C = x.shape


        qkv = self.qkv(x)


        q,k,v = torch.chunk(
            qkv,
            3,
            dim=-1
        )


        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )


        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


        scores = (
            q @ k.transpose(-2,-1)
        ) / (self.head_dim ** 0.5)


        scores = scores.masked_fill(
            self.mask[:T,:T] == 0,
            float("-inf")
        )


        weights = torch.softmax(
            scores,
            dim=-1
        )


        weights = self.dropout(weights)


        out = weights @ v


        out = out.transpose(
            1,
            2
        ).contiguous()


        out = out.view(
            B,
            T,
            C
        )


        return self.proj(out)