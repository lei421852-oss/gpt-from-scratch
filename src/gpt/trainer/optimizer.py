import torch


def build_adamw(
    model,
    learning_rate,
    weight_decay,
    betas=(0.9, 0.95),
):
    """Build AdamW with decay applied only to matrix-like parameters."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")

    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        target_group = (
            decay_parameters
            if parameter.ndim >= 2
            else no_decay_parameters
        )
        target_group.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
    )
