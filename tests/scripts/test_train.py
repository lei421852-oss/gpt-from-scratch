import importlib.util
from pathlib import Path

import pytest

TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
SPEC = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
TRAIN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_MODULE)


def test_split_tokens_keeps_splits_separate():
    train_tokens, validation_tokens = TRAIN_MODULE.split_tokens(
        list(range(20)),
        validation_split=0.25,
        block_size=2,
    )

    assert train_tokens == list(range(15))
    assert validation_tokens == list(range(15, 20))


@pytest.mark.parametrize("validation_split", [0, 1, -0.1, 1.1])
def test_split_tokens_rejects_invalid_ratio(validation_split):
    with pytest.raises(ValueError):
        TRAIN_MODULE.split_tokens(
            list(range(20)),
            validation_split=validation_split,
            block_size=2,
        )


def test_validation_loader_samples_across_entire_dataset():
    dataset = list(range(100))

    loader = TRAIN_MODULE.build_validation_loader(
        dataset,
        batch_size=2,
        eval_batches=3,
    )
    sampled_values = [value.item() for batch in loader for value in batch]

    assert len(sampled_values) == 6
    assert sampled_values[0] == 0
    assert sampled_values[-1] == 99
