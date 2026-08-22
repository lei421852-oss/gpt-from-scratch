import math


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        max_learning_rate,
        min_learning_rate,
        warmup_steps,
        max_steps,
    ):
        if max_learning_rate <= 0:
            raise ValueError("max_learning_rate must be greater than zero")
        if not 0 <= min_learning_rate <= max_learning_rate:
            raise ValueError(
                "min_learning_rate must be between zero and max_learning_rate"
            )
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if max_steps <= warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")

        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_number = 0
        self._set_learning_rate(self.learning_rate_at(0))

    def learning_rate_at(self, step):
        if step < 0:
            raise ValueError("step cannot be negative")

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.max_learning_rate * step / self.warmup_steps

        if step >= self.max_steps:
            return self.min_learning_rate

        decay_ratio = (
            (step - self.warmup_steps)
            / (self.max_steps - self.warmup_steps)
        )
        cosine = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return self.min_learning_rate + cosine * (
            self.max_learning_rate - self.min_learning_rate
        )

    def _set_learning_rate(self, learning_rate):
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def step(self, step=None):
        self.step_number = self.step_number + 1 if step is None else step
        learning_rate = self.learning_rate_at(self.step_number)
        self._set_learning_rate(learning_rate)
        return learning_rate

    def state_dict(self):
        return {"step_number": self.step_number}

    def load_state_dict(self, state_dict):
        self.step_number = state_dict["step_number"]
        self._set_learning_rate(self.learning_rate_at(self.step_number))
