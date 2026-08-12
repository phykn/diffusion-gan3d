import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.evaluate import percolating_fractions

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "scripts" / "paper"
PAPER_SCRIPTS = (
    "make_reference.py",
    "make_anchor_asset.py",
    "make_scale_up_asset.py",
    "evaluate_anchor_sweep.py",
    "evaluate_overlap_ablation.py",
    "evaluate_paper_metrics.py",
)


def _tree(filename: str) -> ast.Module:
    path = PAPER_DIR / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _load_script(filename: str):
    path = PAPER_DIR / filename
    name = f"_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load paper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PAPER_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PAPER_DIR))
    return module


def _arguments(tree: ast.Module) -> dict[str, ast.Call]:
    arguments = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        option = node.args[0]
        if isinstance(option, ast.Constant) and isinstance(option.value, str):
            arguments[option.value] = node
    return arguments


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def test_paper_metrics_reports_mean_pore_percolation() -> None:
    module = _load_script("evaluate_paper_metrics.py")
    volume = np.zeros((4, 4, 4), dtype=np.uint8)

    expected = float(np.mean(percolating_fractions(volume, 0)))
    assert module.PORE_PHASE == 0
    assert module.percolating_fractions is percolating_fractions
    assert module.mean_percolation(volume) == pytest.approx(expected)


def test_anchor_sweep_pore_output_contract() -> None:
    source = (PAPER_DIR / "evaluate_anchor_sweep.py").read_text(encoding="utf-8")

    for field in ("percolation", "generation_seconds"):
        assert f'"{field}"' in source
    assert '"pore_percolation": {' in source
    assert '"Percolation, phase 0 (%)"' in source
    assert "percolation_error" not in source


def test_paper_metrics_reuses_real_fid_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("evaluate_paper_metrics.py")
    metric = object()
    scored: list[np.ndarray] = []

    def fake_make_fid_metric(real: np.ndarray, device: torch.device):
        assert np.array_equal(real, np.asarray([-1.0]))
        assert device.type == "cpu"
        return metric

    def fake_fid_score(
        received: object,
        values: np.ndarray,
        device: torch.device,
    ) -> float:
        assert received is metric
        assert device.type == "cpu"
        values = np.asarray(values)
        scored.append(values)
        return float(values.mean())

    monkeypatch.setattr(module, "make_fid_metric", fake_make_fid_metric)
    monkeypatch.setattr(module, "fid_score", fake_fid_score)
    monkeypatch.setattr(module, "REAL_EVALUATION_SEEDS", (10,))
    monkeypatch.setattr(module, "CONDITIONS", ("Real 2D crops", "Generated"))
    monkeypatch.setattr(module, "SEEDS", (20, 21))
    monkeypatch.setattr(
        module,
        "sample_real_crops",
        lambda seed, _output_size: np.asarray([float(seed)]),
    )
    monkeypatch.setattr(
        module,
        "volume_path",
        lambda condition, seed: (condition, seed),
    )
    monkeypatch.setattr(
        module,
        "load_volume",
        lambda path: np.asarray([float(path[1])]),
    )
    monkeypatch.setattr(
        module,
        "select_metric_slices",
        lambda volume, **_kwargs: volume,
    )
    monkeypatch.setattr(module, "phase_fraction", lambda *_args: 0.5)

    real_rows, reference_fid, generated = module.compute_fid_scores(
        np.asarray([-1.0]),
        np.asarray([99.0]),
        patch_size=4,
        device=torch.device("cpu"),
    )

    assert [values.tolist() for values in scored] == [
        [10.0],
        [99.0],
        [20.0],
        [21.0],
    ]
    assert real_rows[0]["fid"] == 10.0
    assert reference_fid == 99.0
    assert set(generated) == {("Generated", 20), ("Generated", 21)}
    assert generated[("Generated", 20)] == pytest.approx(20.0)
    assert generated[("Generated", 21)] == pytest.approx(21.0)


@pytest.mark.parametrize("filename", PAPER_SCRIPTS)
def test_paper_generation_cli_requires_weight_and_loads_guidance_from_yaml(
    filename: str,
) -> None:
    tree = _tree(filename)
    arguments = _arguments(tree)

    weight = _keywords(arguments["--weight"])
    assert isinstance(weight["required"], ast.Constant)
    assert weight["required"].value is True

    guidance = _keywords(arguments["--guidance"])
    assert isinstance(guidance["type"], ast.Name)
    assert guidance["type"].id == "float"
    assert "default" not in guidance

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "load_generation_settings" in imported_names
    assert "find_weights" not in imported_names


@pytest.mark.parametrize("filename", PAPER_SCRIPTS)
def test_every_paper_generation_call_forwards_guidance(filename: str) -> None:
    calls = [
        node
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"generate", "generate_probs"}
    ]

    assert calls
    for call in calls:
        assert "guidance" in _keywords(call)
        assert "margin" in _keywords(call)


@pytest.mark.parametrize("filename", PAPER_SCRIPTS)
def test_paper_cli_builds_common_provenance(filename: str) -> None:
    calls = [
        node
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_provenance"
    ]

    assert len(calls) == 1


@pytest.mark.parametrize(
    "filename",
    ("evaluate_anchor_sweep.py", "evaluate_paper_metrics.py"),
)
def test_generation_helper_accepts_guidance_explicitly(filename: str) -> None:
    function = next(
        node
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.FunctionDef) and node.name == "generate_volumes"
    )

    assert "guidance" in {argument.arg for argument in function.args.args}


def test_common_manifest_rejects_changed_inputs_and_cache_paths(tmp_path: Path) -> None:
    module = _load_script("provenance.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights = run_dir / "generator.pt"
    weights.write_bytes(b"weights-v1")
    config = run_dir / "train.yaml"
    config.write_text("train: {}\n", encoding="utf-8")
    reference = tmp_path / "reference.tiff"
    reference.write_bytes(b"reference-v1")
    cached = tmp_path / "cached.tiff"
    cached.write_bytes(b"cached-v1")
    manifest = tmp_path / "manifest.json"
    provenance = module.build_provenance(
        weights,
        1.0,
        generation={"blocks": (2, 2, 2)},
        reference=reference,
    )
    data = {
        **provenance,
        "cached_outputs": module.describe_files((cached,)),
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")

    assert provenance["generation"]["blocks"] == [2, 2, 2]
    assert data["cached_outputs"] == [str(cached.resolve())]
    module.validate_manifest(
        manifest,
        provenance,
        label="test reuse",
        cached_paths=(cached,),
    )

    weights.write_bytes(b"weights-v2")
    changed = module.build_provenance(
        weights,
        1.0,
        generation={"blocks": (2, 2, 2)},
        reference=reference,
    )
    with pytest.raises(ValueError, match="weight_sha256"):
        module.validate_manifest(manifest, changed, label="test reuse")
    weights.write_bytes(b"weights-v1")

    reference.write_bytes(b"reference-v2")
    changed = module.build_provenance(
        weights,
        1.0,
        generation={"blocks": (2, 2, 2)},
        reference=reference,
    )
    with pytest.raises(ValueError, match="reference_sha256"):
        module.validate_manifest(manifest, changed, label="test reuse")
    reference.write_bytes(b"reference-v1")

    config.write_text("train:\n  changed: true\n", encoding="utf-8")
    changed = module.build_provenance(
        weights,
        1.0,
        generation={"blocks": (2, 2, 2)},
        reference=reference,
    )
    with pytest.raises(ValueError, match="train_config_sha256"):
        module.validate_manifest(manifest, changed, label="test reuse")
    config.write_text("train: {}\n", encoding="utf-8")

    cached.write_bytes(b"cached-v2")
    module.validate_manifest(
        manifest,
        provenance,
        label="test reuse",
        cached_paths=(cached,),
    )
    other = tmp_path / "other.tiff"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="cached output paths"):
        module.validate_manifest(
            manifest,
            provenance,
            label="test reuse",
            cached_paths=(other,),
        )
    cached.unlink()
    with pytest.raises(FileNotFoundError, match="cached output was not found"):
        module.validate_manifest(
            manifest,
            provenance,
            label="test reuse",
            cached_paths=(cached,),
        )


def test_provenance_rejects_output_collision(tmp_path: Path) -> None:
    module = _load_script("provenance.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights = run_dir / "generator.pt"
    weights.write_bytes(b"weights")
    (run_dir / "train.yaml").write_text("train: {}\n", encoding="utf-8")
    provenance = module.build_provenance(
        weights,
        1.0,
        generation={"seed": 0},
    )

    with pytest.raises(ValueError, match="conflicts with an input"):
        module.validate_output_paths(provenance, (weights,))


def test_overlap_metrics_report_degenerate_outputs_without_crashing() -> None:
    module = _load_script("evaluate_overlap_ablation.py")
    volume = torch.zeros((16, 8, 8), dtype=torch.uint8)

    assert math.isnan(module.exact_seam_change_ratio(volume, seam=8))
    assert math.isnan(module.optional_float(None))


def test_paper_metrics_uses_fixed_block_geometry() -> None:
    module = _load_script("evaluate_paper_metrics.py")

    assert module.scale_output_shape(128, 8) == (352, 352, 352)


@pytest.mark.parametrize(
    "filename",
    ("evaluate_anchor_sweep.py", "evaluate_paper_metrics.py"),
)
def test_reuse_cli_validates_all_cached_outputs(filename: str) -> None:
    calls = [
        node
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_manifest"
    ]

    assert len(calls) == 1
    assert "cached_paths" in _keywords(calls[0])


def test_anchor_asset_writes_generation_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("make_anchor_asset.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weight = run_dir / "generator.pt"
    weight.write_bytes(b"anchor-weights")
    (run_dir / "train.yaml").write_text(
        "train: {}\n",
        encoding="utf-8",
    )
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample-image")

    class FakeGenerator:
        patch_size = 4

        def generate(self, *, anchors, guidance, domain, margin):
            assert len(anchors) == 1
            assert guidance == 1.5
            assert domain == 0
            assert margin == 8
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SAMPLE_PATH", sample)
    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module,
        "load_center_roi",
        lambda _size: torch.zeros((4, 4), dtype=torch.long),
    )
    monkeypatch.setattr(
        module,
        "render_result",
        lambda _anchor, _generated, _volume, output: output.write_bytes(b"png"),
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "make_anchor_asset.py",
            "--weight",
            str(weight),
            "--guidance",
            "1.5",
        ],
    )

    module.main()

    metadata = json.loads(
        (tmp_path / "04-anchor-conditioning.json").read_text(encoding="utf-8")
    )
    assert metadata["weights"] == str(weight.resolve())
    assert len(metadata["weight_sha256"]) == 64
    assert len(metadata["train_config_sha256"]) == 64
    assert metadata["guidance"] == 1.5
    assert len(metadata["reference_sha256"]) == 64
    assert len(metadata["generation_signature"]) == 64
    assert metadata["seed"] == module.SEED
    assert metadata["generation"]["domain"] == 0
    assert metadata["anchor"]["axis"] == module.AXIS
    assert metadata["output"]["shape"] == [4, 4, 4]


def test_scale_asset_writes_generation_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("make_scale_up_asset.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weight = run_dir / "generator.pt"
    weight.write_bytes(b"scale-weights")
    (run_dir / "train.yaml").write_text(
        "train: {}\n",
        encoding="utf-8",
    )
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample-image")

    class FakeGenerator:
        patch_size = 4

        def generate(self, *, anchors, guidance, domain, margin):
            assert len(anchors) == 1
            assert guidance == 1.75
            assert domain == 0
            assert margin == 0
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    class FakeScaled:
        def shape_from_blocks(self, blocks, overlap):
            assert blocks == module.BLOCKS
            assert overlap == 0
            return (12, 12, 12)

        def plan(self, shape, overlap):
            assert shape == (12, 12, 12)
            assert overlap == 0
            return SimpleNamespace(
                tile_count=27,
                base_shell=0,
                tile_size=4,
                stride=4,
                shape=shape,
                seams=((4, 8), (4, 8), (4, 8)),
            )

        def generate(self, **kwargs):
            assert kwargs["guidance"] == 1.75
            assert kwargs["margin"] == 0
            assert kwargs["domain"] == 0
            return torch.zeros((12, 12, 12), dtype=torch.uint8)

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SAMPLE_PATH", sample)
    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module,
        "load_generation_settings",
        lambda: SimpleNamespace(guidance=1.0, overlap=0, margin=0),
    )
    monkeypatch.setattr(module, "ScaledGenerator", lambda _generator: FakeScaled())
    monkeypatch.setattr(
        module,
        "load_center_roi",
        lambda _size: torch.zeros((4, 4), dtype=torch.long),
    )
    monkeypatch.setattr(
        module,
        "render_result",
        lambda **kwargs: kwargs["output"].write_bytes(b"png"),
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "make_scale_up_asset.py",
            "--weight",
            str(weight),
            "--guidance",
            "1.75",
        ],
    )

    module.main()

    metadata = json.loads((tmp_path / "05-scale-up.json").read_text(encoding="utf-8"))
    assert metadata["weights"] == str(weight.resolve())
    assert len(metadata["weight_sha256"]) == 64
    assert len(metadata["train_config_sha256"]) == 64
    assert metadata["guidance"] == 1.75
    assert len(metadata["reference_sha256"]) == 64
    assert len(metadata["generation_signature"]) == 64
    assert metadata["seed"] == module.SEED
    assert metadata["generation"]["blocks"] == [3, 3, 3]
    assert metadata["generation"]["domain"] == 0
    assert metadata["scale_plan"]["tile_count"] == 27
    assert metadata["output"]["shape"] == [12, 12, 12]
