from pathlib import Path

from gpt.utils.config import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_model_config():
    config = load_yaml(PROJECT_ROOT / "configs" / "model.yaml")

    assert config.model.embedding_dim == 256
    assert config.model.num_layers == 6
    assert config.model.num_heads == 8
    assert config.model.dropout == 0.1


def test_config_converts_back_to_nested_dictionary():
    config = load_yaml(PROJECT_ROOT / "configs" / "model.yaml")

    assert config.to_dict()["model"] == {
        "vocab_size": 65,
        "block_size": 128,
        "embedding_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
    }
