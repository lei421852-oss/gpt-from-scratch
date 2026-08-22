from pathlib import Path

import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device, experiment=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.experiment = experiment

    @staticmethod
    def _loss(logits, targets):
        batch_size, sequence_length, vocab_size = logits.shape
        return F.cross_entropy(
            logits.reshape(batch_size * sequence_length, vocab_size),
            targets.reshape(batch_size * sequence_length),
        )

    def train_step(self, x, y):
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = self._loss(logits, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, data_loader, max_batches=None):
        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be greater than zero")

        was_training = self.model.training
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_index, (x, y) in enumerate(data_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)
                losses.append(self._loss(self.model(x), y).item())

        self.model.train(was_training)
        return sum(losses) / len(losses)

    def save_checkpoint(self, path, step):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["step"]

    def _log_metrics(self, metrics):
        if self.experiment is not None:
            self.experiment.log_metrics(metrics)

    def _save_named_checkpoint(self, name, step):
        if self.experiment is not None:
            self.save_checkpoint(self.experiment.checkpoint_path(name), step)

    def train(
        self,
        train_loader,
        max_steps,
        log_interval=100,
        validation_loader=None,
        eval_interval=None,
        eval_batches=None,
        checkpoint_interval=None,
        start_step=0,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if log_interval <= 0:
            raise ValueError("log_interval must be greater than zero")
        if len(train_loader) == 0:
            raise ValueError("train_loader must contain at least one batch")
        if validation_loader is not None and (
            eval_interval is None or eval_interval <= 0
        ):
            raise ValueError("eval_interval must be set with validation_loader")
        if checkpoint_interval is not None and checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than zero")

        step = start_step
        last_loss = None

        while step < max_steps:
            for x, y in train_loader:
                last_loss = self.train_step(x, y)
                step += 1

                if step == 1 or step % log_interval == 0:
                    print(f"step {step}, train loss {last_loss:.4f}")
                    self._log_metrics(
                        {"step": step, "split": "train", "loss": last_loss}
                    )

                if validation_loader is not None and step % eval_interval == 0:
                    validation_loss = self.evaluate(
                        validation_loader,
                        max_batches=eval_batches,
                    )
                    print(
                        f"step {step}, validation loss "
                        f"{validation_loss:.4f}"
                    )
                    self._log_metrics(
                        {
                            "step": step,
                            "split": "validation",
                            "loss": validation_loss,
                        }
                    )

                if checkpoint_interval and step % checkpoint_interval == 0:
                    self._save_named_checkpoint(
                        f"step_{step:08d}.pt",
                        step,
                    )

                if step >= max_steps:
                    break

        self._save_named_checkpoint("last.pt", step)
        return last_loss
