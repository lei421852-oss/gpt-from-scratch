from torch import nn

from gpt.trainer.optimizer import build_adamw


class ModelWithMatrixAndVectorParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.normalization = nn.LayerNorm(4)


def test_build_adamw_partitions_every_parameter_once():
    model = ModelWithMatrixAndVectorParameters()
    optimizer = build_adamw(model, 3e-4, 0.1)

    optimized_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    assert len(optimized_parameters) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in optimized_parameters}) == len(
        optimized_parameters
    )


def test_build_adamw_decays_only_matrix_parameters():
    model = ModelWithMatrixAndVectorParameters()
    optimizer = build_adamw(model, 3e-4, 0.1)
    groups = {
        group["weight_decay"]: group["params"]
        for group in optimizer.param_groups
    }

    assert all(parameter.ndim >= 2 for parameter in groups[0.1])
    assert all(parameter.ndim < 2 for parameter in groups[0.0])
