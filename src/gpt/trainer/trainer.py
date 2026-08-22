import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        experiment=None,
        scheduler=None,
        gradient_clip_norm=None,
        amp=True,
        artifact_metadata=None,
        data_generator=None,
    ):
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be greater than zero")

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.experiment = experiment
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.artifact_metadata = dict(artifact_metadata or {})
        self.data_generator = data_generator
        self.amp_enabled = amp and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.amp_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.scaler_enabled,
        )
        self.last_gradient_norm = None

    @staticmethod
    def _loss(logits, targets):
        batch_size, sequence_length, vocab_size = logits.shape
        return F.cross_entropy(
            logits.reshape(batch_size * sequence_length, vocab_size),
            targets.reshape(batch_size * sequence_length),
        )

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
        )

    def train_step(self, x, y):
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast_context():
            loss = self._loss(self.model(x), y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        max_norm = (
            self.gradient_clip_norm
            if self.gradient_clip_norm is not None
            else float("inf")
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm,
        )
        self.last_gradient_norm = float(gradient_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def evaluate(self, data_loader, max_batches=None):
        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be greater than zero")

        was_training = self.model.training
        self.model.eval()
        losses = []

        try:
            with torch.no_grad():
                for batch_index, (x, y) in enumerate(data_loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break

                    x = x.to(self.device)
                    y = y.to(self.device)
                    with self._autocast_context():
                        loss = self._loss(self.model(x), y)
                    losses.append(loss.item())
        finally:
            self.model.train(was_training)

        return sum(losses) / len(losses)

    def _random_state(self):
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if self.data_generator is not None:
            state["data_generator"] = self.data_generator.get_state()
        return state

    def _restore_random_state(self, state):
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        if "data_generator" in state and self.data_generator is not None:
            self.data_generator.set_state(state["data_generator"].cpu())

    def save_checkpoint(self, path, step):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "artifact_format_version": 1,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "random_state": self._random_state(),
            **self.artifact_metadata,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "random_state" in checkpoint:
            self._restore_random_state(checkpoint["random_state"])
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
                next_step = step + 1
                if self.scheduler is not None:
                    self.scheduler.step(next_step)

                last_loss = self.train_step(x, y)
                step = next_step

                if step == 1 or step % log_interval == 0:
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"step {step}, train loss {last_loss:.4f}, "
                        f"lr {learning_rate:.6g}, "
                        f"grad norm {self.last_gradient_norm:.4f}"
                    )
                    self._log_metrics(
                        {
                            "step": step,
                            "split": "train",
                            "loss": last_loss,
                            "learning_rate": learning_rate,
                            "gradient_norm": self.last_gradient_norm,
                        }
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
