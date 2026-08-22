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

        self.block_size = block_size

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

    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        was_training = self.training
        self.eval()
        generated = input_ids

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    context = generated[:, -self.block_size:]
                    next_token_logits = self(context)[:, -1, :]

                    if temperature == 0:
                        next_token = torch.argmax(
                            next_token_logits,
                            dim=-1,
                            keepdim=True,
                        )
                    else:
                        next_token_logits = next_token_logits / temperature

                        if top_k is not None:
                            k = min(top_k, next_token_logits.shape[-1])
                            top_values, _ = torch.topk(
                                next_token_logits,
                                k,
                            )
                            cutoff = top_values[:, -1].unsqueeze(-1)
                            next_token_logits = next_token_logits.masked_fill(
                                next_token_logits < cutoff,
                                float("-inf"),
                            )

                        probabilities = torch.softmax(
                            next_token_logits,
                            dim=-1,
                        )
                        next_token = torch.multinomial(
                            probabilities,
                            num_samples=1,
                        )

                    generated = torch.cat(
                        (generated, next_token),
                        dim=1,
                    )
        finally:
            self.train(was_training)

        return generated
