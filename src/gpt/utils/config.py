from pathlib import Path
from typing import Any, Mapping

import yaml


class Config:
    """Recursively expose configuration mappings through attributes."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

        for key, value in data.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError(f"Invalid configuration key: {key!r}")

            if isinstance(value, Mapping):
                value = Config(value)

            setattr(self, key, value)


def load_yaml(path: str | Path) -> Config:
    """Load a YAML mapping and return it as a Config object."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        data = {}

    if not isinstance(data, Mapping):
        raise ValueError(
            f"Configuration root must be a mapping: {config_path}"
        )

    return Config(data)

