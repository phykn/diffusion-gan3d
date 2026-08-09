import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def test_paper_metrics_reuses_real_kid_features_and_resets_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("evaluate_paper_metrics.py")

    class FakeKid:
        constructions = 0
        real_updates = 0
        fake_updates = 0
        compute_fake_counts: tuple[int, ...] = ()
        compute_seeds: tuple[int, ...] = ()
        resets = 0
        destroyed = False

        def __init__(self, **kwargs) -> None:
            assert kwargs == {
                "feature": 2048,
                "subsets": module.KID_SUBSETS,
                "subset_size": module.KID_SUBSET_SIZE,
                "reset_real_features": False,
                "normalize": False,
            }
            type(self).constructions += 1
            self.real_features: list[torch.Tensor] = []
            self.fake_features: list[torch.Tensor] = []

        def __del__(self) -> None:
            type(self).destroyed = True

        def to(self, device: torch.device):
            assert device.type == "cpu"
            return self

        def update(self, images: torch.Tensor, *, real: bool) -> None:
            if real:
                type(self).real_updates += 1
                self.real_features.append(images)
            else:
                assert not self.fake_features
                type(self).fake_updates += 1
                self.fake_features.append(images)

        def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
            assert len(self.real_features) == 1
            type(self).compute_fake_counts += (len(self.fake_features),)
            type(self).compute_seeds += (torch.initial_seed(),)
            value = self.fake_features[0].to(torch.float32).mean()
            return value, value / 10.0

        def reset(self) -> None:
            type(self).resets += 1
            self.fake_features.clear()

    monkeypatch.setattr(module, "KernelInceptionDistance", FakeKid)
    monkeypatch.setattr(module, "metric_images", lambda values, _device: values)
    monkeypatch.setattr(module, "REAL_EVALUATION_SEEDS", (10,))
    monkeypatch.setattr(module, "CONDITIONS", ("Real 2D crops", "Generated"))
    monkeypatch.setattr(module, "SEEDS", (20, 21))
    monkeypatch.setattr(
        module,
        "sample_real_crops",
        lambda seed, _output_size: torch.tensor([float(seed)]),
    )
    monkeypatch.setattr(
        module,
        "volume_path",
        lambda condition, seed: (condition, seed),
    )
    monkeypatch.setattr(
        module,
        "load_volume",
        lambda path: torch.tensor([float(path[1])]),
    )
    monkeypatch.setattr(
        module,
        "select_metric_slices",
        lambda volume, **_kwargs: volume,
    )
    monkeypatch.setattr(module, "porosity", lambda _values: 0.5)
    monkeypatch.setattr(
        module,
        "tortuosity",
        lambda *_args, **_kwargs: pytest.fail("TauFactor overlapped the KID pass"),
    )

    with torch.random.fork_rng():
        real_rows, generated = module.compute_kid_scores(
            torch.tensor([-1.0]),
            patch_size=4,
            device=torch.device("cpu"),
        )

    assert FakeKid.constructions == 1
    assert FakeKid.real_updates == 1
    assert FakeKid.fake_updates == 3
    assert FakeKid.compute_fake_counts == (1, 1, 1)
    assert FakeKid.compute_seeds == (0, 0, 0)
    assert FakeKid.resets == 3
    assert FakeKid.destroyed
    assert real_rows[0]["kid"] == 10.0
    assert set(generated) == {("Generated", 20), ("Generated", 21)}
    assert generated[("Generated", 20)] == pytest.approx((20.0, 2.0))
    assert generated[("Generated", 21)] == pytest.approx((21.0, 2.1))


@pytest.mark.parametrize("filename", PAPER_SCRIPTS)
def test_paper_generation_cli_requires_weight_and_defaults_guidance(
    filename: str,
) -> None:
    tree = _tree(filename)
    arguments = _arguments(tree)

    weight = _keywords(arguments["--weight"])
    assert isinstance(weight["required"], ast.Constant)
    assert weight["required"].value is True

    guidance = _keywords(arguments["--guidance-scale"])
    assert isinstance(guidance["type"], ast.Name)
    assert guidance["type"].id == "float"
    assert isinstance(guidance["default"], ast.Constant)
    assert guidance["default"].value == 1.0

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
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
        assert "guidance_scale" in _keywords(call)


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

    assert "guidance_scale" in {argument.arg for argument in function.args.args}


def test_common_manifest_rejects_changed_inputs_and_cache(tmp_path: Path) -> None:
    module = _load_script("provenance.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights = run_dir / "generator.pt"
    weights.write_bytes(b"weights-v1")
    config = run_dir / "train.yaml"
    config.write_text("schema_version: 2\n", encoding="utf-8")
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

    assert provenance["schema_version"] == 2
    assert provenance["generation"]["blocks"] == [2, 2, 2]
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

    config.write_text("schema_version: 2\nchanged: true\n", encoding="utf-8")
    changed = module.build_provenance(
        weights,
        1.0,
        generation={"blocks": (2, 2, 2)},
        reference=reference,
    )
    with pytest.raises(ValueError, match="train_config_sha256"):
        module.validate_manifest(manifest, changed, label="test reuse")
    config.write_text("schema_version: 2\n", encoding="utf-8")

    cached.write_bytes(b"cached-v2")
    with pytest.raises(ValueError, match="cached output SHA-256"):
        module.validate_manifest(
            manifest,
            provenance,
            label="test reuse",
            cached_paths=(cached,),
        )

    config.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version must be 2"):
        module.build_provenance(
            weights,
            1.0,
            generation={"blocks": (2, 2, 2)},
            reference=reference,
        )


def test_provenance_verifies_sources_and_rejects_output_collision(
    tmp_path: Path,
) -> None:
    module = _load_script("provenance.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights = run_dir / "generator.pt"
    weights.write_bytes(b"weights")
    (run_dir / "train.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    source = tmp_path / "sampler.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    provenance = module.build_provenance(
        weights,
        1.0,
        generation={"seed": 0},
        source_files=(source,),
    )

    module.verify_provenance_inputs(provenance)
    with pytest.raises(ValueError, match="conflicts with an input"):
        module.validate_output_paths(provenance, (weights,))

    source.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preflight"):
        module.verify_provenance_inputs(provenance)


def test_overlap_metrics_report_degenerate_outputs_without_crashing() -> None:
    module = _load_script("evaluate_overlap_ablation.py")
    volume = torch.zeros((16, 8, 8), dtype=torch.uint8)

    assert math.isnan(module.exact_seam_change_ratio(volume, seam=8))
    assert math.isnan(module.optional_float(None))


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
        "schema_version: 2\n",
        encoding="utf-8",
    )
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample-image")

    class FakeGenerator:
        patch_size = 4
        anchor_enabled = True

        def generate(self, *, anchors, guidance_scale):
            assert len(anchors) == 1
            assert guidance_scale == 1.5
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
            "--guidance-scale",
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
    assert metadata["schema_version"] == 2
    assert metadata["guidance_scale"] == 1.5
    assert len(metadata["reference_sha256"]) == 64
    assert len(metadata["generation_signature"]) == 64
    assert metadata["seed"] == module.SEED
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
        "schema_version: 2\n",
        encoding="utf-8",
    )
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample-image")

    class FakeGenerator:
        patch_size = 4
        anchor_enabled = True

        def generate(self, *, anchors, guidance_scale):
            assert len(anchors) == 1
            assert guidance_scale == 1.75
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    class FakeScaled:
        def plan(self, shape, overlap):
            assert shape == (12, 12, 12)
            assert overlap == module.OVERLAP
            return SimpleNamespace(tile_count=27, base_shell=0)

        def generate(self, **kwargs):
            assert kwargs["guidance_scale"] == 1.75
            return torch.zeros((12, 12, 12), dtype=torch.uint8)

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SAMPLE_PATH", sample)
    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
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
            "--guidance-scale",
            "1.75",
        ],
    )

    module.main()

    metadata = json.loads((tmp_path / "05-scale-up.json").read_text(encoding="utf-8"))
    assert metadata["weights"] == str(weight.resolve())
    assert len(metadata["weight_sha256"]) == 64
    assert len(metadata["train_config_sha256"]) == 64
    assert metadata["schema_version"] == 2
    assert metadata["guidance_scale"] == 1.75
    assert len(metadata["reference_sha256"]) == 64
    assert len(metadata["generation_signature"]) == 64
    assert metadata["seed"] == module.SEED
    assert metadata["generation"]["blocks"] == [3, 3, 3]
    assert metadata["scale_plan"]["tile_count"] == 27
    assert metadata["output"]["shape"] == [12, 12, 12]
