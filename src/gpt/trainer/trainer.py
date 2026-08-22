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


    def train(
        self,
        data_loader,
        max_steps,
        log_interval=100
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        if log_interval <= 0:
            raise ValueError("log_interval must be greater than zero")

        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch")

        step = 0
        last_loss = None

        while step < max_steps:
            for x, y in data_loader:
                last_loss = self.train_step(x, y)
                step += 1

                if step == 1 or step % log_interval == 0:
                    print(f"step {step}, loss {last_loss:.4f}")

                if step >= max_steps:
                    break

        return last_loss
