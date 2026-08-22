import json

from gpt.utils.experiment import Experiment


def test_experiment_snapshots_configs_and_logs_metrics(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("training:\n  seed: 42\n", encoding="utf-8")

    experiment = Experiment(
        tmp_path / "experiments",
        "test run",
        config_paths=(config_path,),
    )
    experiment.log_metrics({"step": 1, "loss": 2.5})

    assert (experiment.directory / "configs" / "train.yaml").exists()
    assert experiment.checkpoint_path("last.pt").parent.exists()
    metrics = [
        json.loads(line)
        for line in experiment.metrics_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert metrics == [{"step": 1, "loss": 2.5}]
