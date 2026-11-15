import train
from src.train.loop import DONE


def test_train_cli_writes_the_lr_run(tiny_sample, tmp_path):
    train.main(
        [
            str(tiny_sample),
            "--run-dir",
            str(tmp_path / "cli"),
            "--lr-only",
            "--device",
            "cpu",
        ]
    )
    run_dir = tmp_path / "cli-lr"
    for name in ("train.yaml", "generator.pt", DONE, "tensorboard"):
        assert (run_dir / name).exists(), name
    events = list((run_dir / "tensorboard").glob("events.*"))
    assert events
