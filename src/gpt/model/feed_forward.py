import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim,
        dropout=0.1
    ):
        super().__init__()


        hidden_dim = 4 * embedding_dim


        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            ),

            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.network(x)