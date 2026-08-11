import importlib
from pathlib import Path

import pytest
import tifffile
import torch

from src.scale import ScalePlan


def make_plan() -> ScalePlan:
    return ScalePlan(
        shape=(8, 8, 8),
        tile_size=6,
        overlap=1,
        stride=4,
        grid=(2, 2, 2),
        tile_count=8,
        states_bytes=6144,
        fusion_bytes=8192,
        tile_bytes=2592,
        workspace_bytes=4096,
        cuda_bytes=18432,
        output_bytes=512,
        cpu_bytes=18736,
        seams=((4,), (4,), (4,)),
    )


def test_memory_size_is_human_readable() -> None:
    module = importlib.import_module("scripts.04_check_scale_up")

    assert module.format_bytes(96 * 1024**3) == "96.00 GiB"
    assert module.format_bytes(8 * 1024**3) == "8.00 GiB"


@pytest.mark.parametrize("count", (None, "0"))
def test_gt_requires_a_positive_anchor_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    count: str | None,
) -> None:
    module = importlib.import_module("scripts.04_check_scale_up")
    argv = [
        "04_check_scale_up.py",
        "--weight",
        str(tmp_path / "generator.pt"),
        "--gt",
        str(tmp_path / "gt.tiff"),
    ]
    if count is not None:
        argv.extend(("--count", count))
    monkeypatch.setattr(module.sys, "argv", argv)

    with pytest.raises(SystemExit):
        module.main()


def test_main_prints_plan_before_scaled_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("scripts.04_check_scale_up")
    plan = make_plan()
    events = []
    output_path = tmp_path / "nested" / "scaled.tiff"

    class FakeGenerator:
        patch_size = 4
        num_phases = 3

    class FakeScaled:
        stats = None

        def plan(self, shape, overlap):
            events.append(("plan", shape, overlap))
            return plan

        def generate(self, **kwargs):
            events.append(
                (
                    "generate",
                    kwargs["blocks"],
                    kwargs["overlap"],
                    kwargs["guidance_scale"],
                    kwargs["domain"],
                )
            )
            assert "output" not in kwargs
            self.stats = plan
            return torch.zeros(plan.shape, dtype=torch.uint8)

    weights = tmp_path / "run" / "generator.pt"
    scaled = FakeScaled()
    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(module, "ScaledGenerator", lambda _generator: scaled)
    monkeypatch.setattr(module, "show_slices", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "04_check_scale_up.py",
            "--weight",
            str(weights),
            "--domain",
            "1",
            "--blocks",
            "2",
            "2",
            "2",
            "--overlap",
            "1",
            "--out",
            str(output_path),
        ],
    )

    module.main()
    assert output_path.is_file()
    assert tifffile.imread(output_path).shape == plan.shape

    output = capsys.readouterr().out
    assert events == [
        ("plan", (8, 8, 8), 1),
        ("generate", (2, 2, 2), 1, 1.0, 1),
    ]
    assert output.index("State memory") < output.index("Status     : scaling")
    assert "Fusion memory: 8.00 KiB" in output
    assert "CUDA total   : 18.00 KiB" in output
    assert "CPU total    : 18.30 KiB" in output
    assert not (weights.parent / "scaled_4x4x4.tiff").exists()


def test_zero_anchor_strength_uses_unanchored_base_without_gt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("scripts.04_check_scale_up")
    plan = make_plan()
    events = []

    class FakeGenerator:
        patch_size = 4
        num_phases = 3

        def generate(self, *, guidance_scale, domain):
            events.append(("base", guidance_scale, domain))
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    class FakeScaled:
        stats = None

        def plan(self, shape, overlap):
            events.append(("plan", overlap))
            return plan

        def generate(self, **kwargs):
            events.append(
                (
                    "scaled",
                    kwargs["overlap"],
                    kwargs["base"] is not None,
                    kwargs["guidance_scale"],
                    kwargs["domain"],
                )
            )
            self.stats = plan
            return torch.zeros(plan.shape, dtype=torch.uint8)

    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(module, "ScaledGenerator", lambda _generator: FakeScaled())
    monkeypatch.setattr(
        module,
        "show_unanchored_base_result",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "04_check_scale_up.py",
            "--weight",
            str(tmp_path / "generator.pt"),
            "--domain",
            "2",
            "--count",
            "3",
            "--anchor-strength",
            "0",
            "--guidance-scale",
            "1.75",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert events == [
        ("plan", 8),
        ("base", 1.75, 2),
        ("scaled", 8, True, 1.75, 2),
    ]
    assert "Base       : unanchored" in output
    assert "anchor planes" not in output
