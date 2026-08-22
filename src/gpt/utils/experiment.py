import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class Experiment:
    """Manage artifacts produced by one training run."""

    def __init__(self, root_dir, name, config_paths=()):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in name
        ).strip("-")
        safe_name = safe_name or "run"

        self.directory = Path(root_dir) / f"{timestamp}-{safe_name}"
        self.checkpoint_directory = self.directory / "checkpoints"
        self.checkpoint_directory.mkdir(parents=True, exist_ok=False)
        self.metrics_path = self.directory / "metrics.jsonl"
        self.summary_path = self.directory / "summary.json"

        config_directory = self.directory / "configs"
        config_directory.mkdir()
        for config_path in config_paths:
            source = Path(config_path)
            shutil.copy2(source, config_directory / source.name)

    def log_metrics(self, metrics):
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def write_summary(self, summary):
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def checkpoint_path(self, name):
        return self.checkpoint_directory / name
