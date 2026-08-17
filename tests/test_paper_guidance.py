import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "scripts" / "paper"
GENERATION_SCRIPTS = (
    "make_anchor_asset.py",
    "make_scale_up_asset.py",
    "evaluate_structure.py",
    "evaluate_continuation.py",
    "evaluate_anchor_coverage.py",
)


def _tree(filename: str) -> ast.Module:
    path = PAPER_DIR / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _load_script(filename: str):
    path = PAPER_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_test_{path.stem}", path)
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
    result = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            result[node.args[0].value] = node
    return result


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def test_make_assets_requires_an_explicit_generated_volume() -> None:
    arguments = _arguments(_tree("make_assets.py"))
    reference = _keywords(arguments["--reference"])

    assert isinstance(reference["required"], ast.Constant)
    assert reference["required"].value is True


@pytest.mark.parametrize("filename", GENERATION_SCRIPTS)
def test_paper_generation_cli_requires_weight_and_uses_yaml_guidance(
    filename: str,
) -> None:
    tree = _tree(filename)
    arguments = _arguments(tree)
    weight = _keywords(arguments["--weight"])
    guidance = _keywords(arguments["--guidance"])

    assert isinstance(weight["required"], ast.Constant)
    assert weight["required"].value is True
    assert isinstance(guidance["type"], ast.Name)
    assert guidance["type"].id == "float"
    assert "default" not in guidance

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "load_generation_settings" in imported


@pytest.mark.parametrize("filename", GENERATION_SCRIPTS)
def test_paper_generation_forwards_guidance_and_margin(filename: str) -> None:
    calls = [
        node
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]

    assert calls
    for call in calls:
        keywords = _keywords(call)
        assert "guidance" in keywords
        assert "margin" in keywords


@pytest.mark.parametrize("filename", GENERATION_SCRIPTS)
def test_paper_generation_records_common_provenance(filename: str) -> None:
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
    ("evaluate_structure.py", "make_scale_up_asset.py"),
)
def test_fixed_block_paper_scale_has_no_outer_margin(filename: str) -> None:
    module = _load_script(filename)
    assert module.SCALE_MARGIN == 0

    calls = [
        node
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scaled"
    ]
    assert len(calls) == 1
    margin = _keywords(calls[0])["margin"]
    assert isinstance(margin, ast.Name)
    assert margin.id == "SCALE_MARGIN"


def test_structure_interface_density_distinguishes_collection_from_volume() -> None:
    module = _load_script("evaluate_structure.py")
    sections = np.asarray(
        [
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
        ],
        dtype=np.uint8,
    )

    assert module.interface_density(sections, spatial_dimensions=2) == pytest.approx(
        0.5
    )
    assert module.interface_density(sections, spatial_dimensions=3) == pytest.approx(
        2 / 3
    )


def test_structure_fid_uses_192_dimensional_features() -> None:
    module = _load_script("evaluate_structure.py")

    assert module.REAL_CROP_COUNT == 64
    assert module.FID_FEATURE_DIMENSIONS == 192


def test_continuation_pool4_similarity_rewards_coarse_match() -> None:
    module = _load_script("evaluate_continuation.py")
    target = torch.zeros((8, 8), dtype=torch.long)
    generated = target.clone()
    generated[0, 0] = 1

    assert module.pooled_similarity(target, target, 2, pool_size=4) == 1.0
    assert module.pooled_similarity(generated, target, 2, pool_size=4) == pytest.approx(
        63 / 64
    )


def test_anchor_coverage_order_is_nested_and_complete() -> None:
    module = _load_script("evaluate_anchor_coverage.py")
    order = module.nested_plane_order(8)

    assert len(order) == 8
    assert set(order) == set(range(8))
    assert order[0] == 4
    assert set(order[:2]).issubset(order[:4])


def test_paper_numbers_and_assets_match_tracked_results() -> None:
    paper = (ROOT / "PAPER.md").read_text(encoding="utf-8")
    structure = json.loads(
        (ROOT / "assets/paper/structure-results.json").read_text(encoding="utf-8")
    )
    continuation = json.loads(
        (ROOT / "assets/paper/05-boundary-continuation.json").read_text(
            encoding="utf-8"
        )
    )
    coverage = json.loads(
        (ROOT / "assets/paper/04-anchor-coverage.json").read_text(encoding="utf-8")
    )

    direct = structure["summary"][1]
    external = continuation["summary"][2]
    assert f"{direct['fid_mean']:.2f} ± {direct['fid_std']:.2f}" in paper
    assert (
        f"{100 * external['pool4_similarity_mean']:.2f} ± "
        f"{100 * external['pool4_similarity_std']:.2f}%"
    ) in paper
    assert (
        f"{external['smoothness_p95_ratio_mean']:.2f} ± "
        f"{external['smoothness_p95_ratio_std']:.2f}"
    ) in paper
    coverage_by_count = {
        result["anchor_count"]: result["voxel_accuracy"]
        for result in coverage["results"]
    }
    assert f"{100 * coverage_by_count[64]:.2f}%" in paper
    assert f"{100 * coverage_by_count[128]:.2f}%" in paper
    assert "192-dimensional Inception-v3 feature layer" in paper

    assets = re.findall(r'<img src="([^"]+)"', paper)
    assert assets
    assert all((ROOT / asset).is_file() for asset in assets)
    assert not any("paper-metrics" in asset for asset in assets)


def test_paper_sidecars_match_rendered_assets() -> None:
    weight_digests = set()
    for stem in (
        "04-anchor-conditioning",
        "04-anchor-coverage",
        "05-boundary-continuation",
        "06-scale-up",
    ):
        image = ROOT / "assets/paper" / f"{stem}.png"
        metadata = json.loads(
            (ROOT / "assets/paper" / f"{stem}.json").read_text(encoding="utf-8")
        )
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        assert metadata["output"]["sha256"] == digest
        weight_digests.add(metadata["weight_sha256"])

    structure = json.loads(
        (ROOT / "assets/paper/structure-results.json").read_text(encoding="utf-8")
    )
    weight_digests.add(structure["weight_sha256"])
    assert len(weight_digests) == 1


def test_common_manifest_rejects_changed_inputs_and_cache_paths(tmp_path: Path) -> None:
    module = _load_script("provenance.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weights = run_dir / "generator.pt"
    weights.write_bytes(b"weights-v1")
    config = run_dir / "train.yaml"
    config.write_text("train: {}\n", encoding="utf-8")
    cached = tmp_path / "cached.tiff"
    cached.write_bytes(b"cached-v1")
    manifest = tmp_path / "manifest.json"
    provenance = module.build_provenance(
        weights,
        1.0,
        generation={"blocks": (2, 2, 2)},
    )
    manifest.write_text(
        json.dumps(
            {
                **provenance,
                "cached_outputs": module.describe_files((cached,)),
            }
        ),
        encoding="utf-8",
    )

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
    )
    with pytest.raises(ValueError, match="weight_sha256"):
        module.validate_manifest(manifest, changed, label="test reuse")

    other = tmp_path / "other.tiff"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="cached output paths"):
        module.validate_manifest(
            manifest,
            provenance,
            label="test reuse",
            cached_paths=(other,),
        )


def test_anchor_asset_writes_generation_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("make_anchor_asset.py")
    weight, sample = _paper_inputs(tmp_path)

    class FakeGenerator:
        patch_size = 4
        default_margin = 8

        def generate(self, *, anchors, anchor_strength, guidance, domain, margin):
            assert len(anchors) == 1
            assert anchor_strength == module.DEFAULT_ANCHOR_STRENGTH
            assert (guidance, domain, margin) == (1.5, 0, 8)
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SAMPLE_PATH", sample)
    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module, "load_center_roi", lambda _size: torch.zeros((4, 4), dtype=torch.long)
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
        ["make_anchor_asset.py", "--weight", str(weight), "--guidance", "1.5"],
    )

    module.main()

    metadata = json.loads(
        (tmp_path / "04-anchor-conditioning.json").read_text(encoding="utf-8")
    )
    assert metadata["weights"] == str(weight.resolve())
    assert metadata["anchor_match"]["accuracy"] == 1.0
    assert metadata["output"]["shape"] == [4, 4, 4]


def test_scale_asset_writes_renumbered_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("make_scale_up_asset.py")
    weight, sample = _paper_inputs(tmp_path)

    class FakeGenerator:
        patch_size = 4
        default_margin = 8

        def generate(self, *, anchors, anchor_strength, guidance, domain, margin):
            assert len(anchors) == 1
            assert anchor_strength == module.DEFAULT_ANCHOR_STRENGTH
            assert (guidance, domain, margin) == (1.75, 0, 8)
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    class FakeScaled:
        def shape_from_blocks(self, blocks, overlap):
            assert blocks == module.BLOCKS
            return (12, 12, 12)

        def plan(self, shape, overlap):
            return SimpleNamespace(
                tile_count=27,
                base_shell=0,
                tile_size=4,
                stride=4,
                shape=shape,
                seams=((4, 8), (4, 8), (4, 8)),
            )

        def generate(self, **kwargs):
            assert kwargs["margin"] == 0
            return torch.zeros((12, 12, 12), dtype=torch.uint8)

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SAMPLE_PATH", sample)
    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module,
        "load_generation_settings",
        lambda: SimpleNamespace(guidance=1.0, overlap=0),
    )
    monkeypatch.setattr(module, "ScaledGenerator", lambda _generator: FakeScaled())
    monkeypatch.setattr(
        module, "load_center_roi", lambda _size: torch.zeros((4, 4), dtype=torch.long)
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
        ["make_scale_up_asset.py", "--weight", str(weight), "--guidance", "1.75"],
    )

    module.main()

    metadata = json.loads((tmp_path / "06-scale-up.json").read_text(encoding="utf-8"))
    assert metadata["weights"] == str(weight.resolve())
    assert metadata["scale_plan"]["tile_count"] == 27
    assert metadata["output"]["shape"] == [12, 12, 12]


def _paper_inputs(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    weight = run_dir / "generator.pt"
    weight.write_bytes(b"weights")
    (run_dir / "train.yaml").write_text("train: {}\n", encoding="utf-8")
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample")
    return weight, sample
