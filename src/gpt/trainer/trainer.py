import torch
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device


    def train_step(
        self,
        x,
        y
    ):

        self.model.train()


        x = x.to(self.device)
        y = y.to(self.device)


        logits = self.model(x)


        B,T,V = logits.shape


        loss = F.cross_entropy(
            logits.view(B*T, V),
            y.view(B*T)
        )


        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


        return loss.item()