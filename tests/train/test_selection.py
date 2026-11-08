"""Best-step selection."""

import yaml

from src.config.schema import update
from src.export import read_export
from src.train.run import train_stage


def _train(config, path, steps=40, every=10):
    config = update(config, "train.total_steps", steps)
    config = update(config, "train.weights_every_steps", every)
    config = update(config, "train.structure_every_steps", 0)
    return train_stage(config, "cpu", path)


def test_generator_is_the_best_scored_step(tiny_configs, tmp_path):
    run = _train(tiny_configs["lr"], tmp_path / "selected")
    scores = yaml.safe_load((run / "selection" / "scores.yaml").read_text())
    assert [entry["step"] for entry in scores["steps"]] == [10, 20, 30, 40]
    for entry in scores["steps"]:
        assert set(entry) == {
            "step",
            "two_point_mae",
            "chord_tv",
            "cluster_error",
            "structure",
        }
    best = scores["best_step"]
    assert best == min(scores["steps"], key=lambda e: e["structure"])["step"]
    assert read_export(run, "lr")["step"] == best
    kept = sorted(path.name for path in (run / "selection").glob("generator_*.pt"))
    assert kept == [f"generator_{best:08d}.pt"]


def test_without_selection_the_last_step_is_exported(tiny_configs, tmp_path):
    config = update(tiny_configs["lr"], "selection.enabled", False)
    run = _train(config, tmp_path / "last", steps=4, every=2)
    assert read_export(run, "lr")["step"] == 4
    assert not (run / "selection").exists()
