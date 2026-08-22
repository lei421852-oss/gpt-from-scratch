from pathlib import Path

from gpt.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_model_config():
    config = load_yaml(PROJECT_ROOT / "configs" / "model.yaml")

    assert config.model.embedding_dim == 256
    assert config.model.num_layers == 6
    assert config.model.num_heads == 8
    assert config.model.dropout == 0.1
