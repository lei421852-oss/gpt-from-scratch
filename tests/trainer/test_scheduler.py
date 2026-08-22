import pytest
import torch

from gpt.trainer.scheduler import WarmupCosineScheduler


def make_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_steps=2,
        max_steps=6,
    )
    return optimizer, scheduler


def test_scheduler_warms_up_then_decays():
    _, scheduler = make_scheduler()
    learning_rates = [scheduler.step(step) for step in range(1, 7)]

    assert learning_rates[0] == pytest.approx(0.5)
    assert learning_rates[1] == pytest.approx(1.0)
    assert learning_rates[2] < learning_rates[1]
    assert learning_rates[-1] == pytest.approx(0.1)
    assert learning_rates == sorted(learning_rates[:2]) + sorted(
        learning_rates[2:], reverse=True
    )


def test_scheduler_restores_step_and_learning_rate():
    _, scheduler = make_scheduler()
    scheduler.step(4)

    restored_optimizer, restored_scheduler = make_scheduler()
    restored_scheduler.load_state_dict(scheduler.state_dict())

    assert restored_scheduler.step_number == 4
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        scheduler.learning_rate_at(4)
    )
