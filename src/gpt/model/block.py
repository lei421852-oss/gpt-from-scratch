from torch import nn

from gpt.model.feed_forward import FeedForward
from gpt.model.multi_head_attention import MultiHeadCausalSelfAttention


class GPTBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()


        self.ln1 = nn.LayerNorm(
            embedding_dim
        )


        self.attention = (
            MultiHeadCausalSelfAttention(
                embedding_dim,
                num_heads,
                block_size,
                dropout
            )
        )


        self.ln2 = nn.LayerNorm(
            embedding_dim
        )


        self.feed_forward = (
            FeedForward(
                embedding_dim,
                dropout
            )
        )


    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )


        x = x + self.feed_forward(
            self.ln2(x)
        )


        return x