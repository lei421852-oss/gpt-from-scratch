import pytest
import torch

from gpt.artifacts import load_model_artifact
from gpt.data.tokenizer import CharacterTokenizer
from gpt.model.gpt import GPT


def make_artifact(path):
    tokenizer = CharacterTokenizer("abc")
    config = {
        "vocab_size": 3,
        "block_size": 4,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
    }
    model = GPT(**config)
    torch.save(
        {
            "artifact_format_version": 1,
            "model_config": config,
            "tokenizer": tokenizer.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return model, tokenizer


def test_load_artifact_requires_no_corpus_or_yaml(tmp_path):
    artifact_path = tmp_path / "model.pt"
    expected_model, expected_tokenizer = make_artifact(artifact_path)

    model, tokenizer, checkpoint = load_model_artifact(
        artifact_path,
        torch.device("cpu"),
    )

    assert checkpoint["artifact_format_version"] == 1
    assert tokenizer.to_dict() == expected_tokenizer.to_dict()
    for expected, actual in zip(
        expected_model.parameters(),
        model.parameters(),
    ):
        assert torch.equal(expected, actual)


def test_load_artifact_rejects_legacy_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="not a portable model artifact"):
        load_model_artifact(
            checkpoint_path,
            torch.device("cpu"),
        )
