"""HTTP API on the tiny SR run."""

import dataclasses
import errno
import tempfile
from io import BytesIO
from pathlib import Path

import anyio
import pytest
import tifffile
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.backend.src import app as server_module
from app.backend.src import response as response_module
from app.backend.src.app import create_app
from app.backend.src.config import load_server_config
from app.backend.src.pipeline import HighResolutionPipeline
from app.backend.src.schema import GenerateRequest

CONFIG = load_server_config(
    Path(__file__).resolve().parents[1] / "fixtures/config/backend.yaml"
)


class Recorder:
    """A pipeline that records what each request asks it to generate."""

    def __init__(self, pipeline: HighResolutionPipeline) -> None:
        self.pipeline = pipeline
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        return self.pipeline.generate(**kwargs)


def tiny_pipeline(run: Path) -> HighResolutionPipeline:
    pipeline = HighResolutionPipeline(run, device="cpu")
    # config/gen.yaml's overlap 8 cannot tile the tiny 16-voxel LR patch.
    pipeline.lr.settings = dataclasses.replace(pipeline.lr.settings, overlap=4)
    return pipeline


@pytest.fixture(scope="session")
def pipeline(tiny_sr_run) -> HighResolutionPipeline:
    return tiny_pipeline(tiny_sr_run)


@pytest.fixture
def service(pipeline) -> Recorder:
    return Recorder(pipeline)


@pytest.fixture
def client(service, tmp_path, monkeypatch) -> TestClient:
    front = tmp_path / "dist"
    assets = front / "assets"
    assets.mkdir(parents=True)
    (front / "index.html").write_text(
        '<div id="app"></div><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("", encoding="utf-8")
    monkeypatch.setattr(server_module, "FRONTEND_DIR", front)
    monkeypatch.setattr(response_module, "pore_tortuosity", lambda _volume, device: 1.5)
    return TestClient(create_app(inference=service, config=CONFIG))


@pytest.fixture
def tiff_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(
        response_module,
        "NamedTemporaryFile",
        lambda **kwargs: tempfile.NamedTemporaryFile(dir=tmp_path, **kwargs),
    )
    return tmp_path


def test_health_reports_the_loaded_model(client, pipeline):
    response = client.get("/api/health")
    assert response.status_code == 200
    # LR blocks of 16 voxels, stride 8, doubled by SR: 2 * (16 + 62 * 8) = 1024.
    assert response.json() == {
        "device": "cpu",
        "crop_size": 32,
        "num_phases": pipeline.num_phases,
        "height_enabled": False,
        "height_extent": None,
        "pore_phase": 0,
        "max_axis_blocks": 63,
    }


def test_health_reports_the_height_range(tiny_anode_sr_run):
    app = create_app(inference=tiny_pipeline(tiny_anode_sr_run), config=CONFIG)
    info = TestClient(app).get("/api/health").json()
    assert info["height_enabled"] is True and info["height_extent"] == 64


def test_frontend_is_served_by_the_api_process(client):
    page = client.get("/")
    assert page.status_code == 200
    assert '<div id="app"></div>' in page.text
    assert "/assets/" in page.text


def test_generate_returns_a_tiff_volume(client, service):
    response = client.post("/api/generate", json={"seed": 3, "tortuosity": True})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert response.headers["x-volume-shape"] == "32,32,32"
    assert response.headers["x-tortuosity"] == "1.5"
    volume = tifffile.imread(BytesIO(response.content))
    assert volume.shape == (32, 32, 32) and volume.dtype.name == "uint8"
    assert service.calls[0]["seed"] == 3
    assert service.calls[0]["anchors"] == ()


def test_the_same_seed_gives_the_same_volume(client):
    first = client.post("/api/generate", json={"seed": 5, "format": "raw"})
    second = client.post("/api/generate", json={"seed": 5, "format": "raw"})
    assert first.content == second.content


def test_generate_forwards_a_profile_and_height(tiny_anode_sr_run):
    service = Recorder(tiny_pipeline(tiny_anode_sr_run))
    client = TestClient(create_app(inference=service, config=CONFIG))
    spec = {
        "axis": "z",
        "points": [[0, [0.2, 0.3, 0.5]], [1, [0.6, 0.2, 0.2]]],
        "interpolation": "linear",
    }
    response = client.post(
        "/api/generate",
        json={"vf_profile": spec, "height_origin": 16, "format": "raw"},
    )
    assert response.status_code == 200
    assert service.calls[0]["vf_profile"] == {
        "axis": "z",
        "points": [(0.0, [0.2, 0.3, 0.5]), (1.0, [0.6, 0.2, 0.2])],
        "interpolation": "linear",
    }
    response = client.post("/api/generate", json={"height_origin": 48})
    assert response.status_code == 422
    assert "outside the source image height" in response.json()["detail"]


def test_generate_can_return_raw_labels(client):
    response = client.post("/api/generate", json={"format": "raw"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert len(response.content) == 32**3


def test_raw_blocks_match_the_volume_shape_header(client):
    response = client.post("/api/generate", json={"blocks": [1, 2, 2], "format": "raw"})
    assert response.status_code == 200
    shape = tuple(map(int, response.headers["x-volume-shape"].split(",")))
    # Two LR blocks of 16 sharing 2 * 4 voxels: 24, then x2.
    assert shape == (32, 48, 48)
    assert len(response.content) == 32 * 48 * 48


@pytest.mark.parametrize("seed", [2**64, 2**100, True, 1.5])
def test_generate_rejects_an_invalid_seed_before_inference(client, service, seed):
    assert client.post("/api/generate", json={"seed": seed}).status_code == 422
    assert not service.calls


def test_generate_accepts_the_largest_torch_seed(client, service):
    response = client.post("/api/generate", json={"seed": 2**64 - 1, "format": "raw"})
    assert response.status_code == 200
    assert service.calls[0]["seed"] == 2**64 - 1


def test_tortuosity_is_opt_in_and_stays_under_the_generation_lock(client, monkeypatch):
    calls = []

    def measure(volume, device):
        assert client.app.state.generate_lock.locked()
        calls.append(volume.shape)
        return None

    monkeypatch.setattr(response_module, "pore_tortuosity", measure)
    response = client.post("/api/generate", json={"format": "raw"})
    assert response.status_code == 200 and not calls
    assert "x-tortuosity" not in response.headers
    response = client.post("/api/generate", json={"format": "raw", "tortuosity": True})
    assert len(calls) == 1 and response.headers["x-tortuosity"] == "unavailable"
    assert not client.app.state.generate_lock.locked()


def test_a_busy_request_is_rejected_without_waiting(client, service):
    lock = client.app.state.generate_lock
    lock.acquire()
    try:
        response = client.post("/api/generate", json={"format": "raw"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert lock.locked()
        assert not service.calls
    finally:
        lock.release()
    assert client.post("/api/generate", json={"format": "raw"}).status_code == 200


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (MemoryError("serialize allocation"), 413),
        (OSError(errno.ENOSPC, "disk full"), 507),
        (OSError(errno.EACCES, "cannot write"), 500),
        (ValueError("encoder rejected the volume"), 500),
    ],
)
def test_response_preparation_errors_clean_up_and_release_the_lock(
    client, monkeypatch, tiff_dir, failure, status
):
    original = response_module.tifffile.imwrite

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(response_module.tifffile, "imwrite", fail)
    response = client.post("/api/generate", json={})
    assert response.status_code == status
    assert not list(tiff_dir.glob("*.tiff"))
    assert not client.app.state.generate_lock.locked()
    monkeypatch.setattr(response_module.tifffile, "imwrite", original)
    assert client.post("/api/generate", json={}).status_code == 200


def test_tiff_response_removes_its_temporary_file(client, tiff_dir):
    response = client.post("/api/generate", json={})
    assert response.status_code == 200
    assert tifffile.imread(BytesIO(response.content)).shape == (32, 32, 32)
    assert not list(tiff_dir.glob("*.tiff"))


def test_a_centered_anchor_and_blocks_reach_the_generator(client, service):
    response = client.post(
        "/api/generate",
        json={
            "anchors": [
                {"image": [[0, 1], [1, 0]], "axis": 0, "index": 0, "position": None}
            ],
            "blocks": [2, 2, 2],
            "anchor_strength": 0.8,
            "format": "raw",
        },
    )
    assert response.status_code == 200
    assert response.headers["x-volume-shape"] == "48,48,48"
    call = service.calls[0]
    assert call["blocks"] == (2, 2, 2)
    anchor = call["anchors"][0]
    assert (anchor.axis, anchor.index, anchor.position) == (0, 0, None)
    assert anchor.image.tolist() == [[0, 1], [1, 0]]
    assert call["anchor_strength"] == 0.8


def test_a_prepared_crop_anchors_the_first_section(client, pipeline):
    image = (torch.arange(32 * 32).reshape(32, 32) // 7 % 2).tolist()
    prepared = client.post("/api/prepare", json={"image": image})
    assert prepared.status_code == 200
    fractions = prepared.json()["image"]
    assert len(fractions) == pipeline.num_phases and len(fractions[0]) == 16
    response = client.post(
        "/api/generate",
        json={
            "anchors": [{"image": fractions, "axis": 0, "index": 0}],
            "blocks": [1, 2, 2],
            "format": "raw",
        },
    )
    assert response.status_code == 200


def test_prepare_rejects_a_crop_of_the_wrong_size(client):
    response = client.post("/api/prepare", json={"image": [[0, 1], [1, 0]]})
    assert response.status_code == 422
    assert "32 x 32" in response.json()["detail"]


def test_generate_rejects_a_non_rectangular_anchor(client):
    response = client.post(
        "/api/generate",
        json={"anchors": [{"image": [[0, 1], [1]], "axis": 0, "index": 0}]},
    )
    assert response.status_code == 422


def test_generate_reports_inference_validation(client):
    response = client.post("/api/generate", json={"blocks": 2, "shape": 16})
    assert response.status_code == 422
    assert "only one of blocks, shape and size" in response.json()["detail"]


def test_generate_reports_a_memory_error(client, service, monkeypatch):
    def reject(**_kwargs):
        raise MemoryError("planned volume does not fit in RAM")

    monkeypatch.setattr(service, "generate", reject)
    response = client.post("/api/generate", json={"blocks": 4})
    assert response.status_code == 413
    assert response.json()["detail"] == (
        "generation exceeds available memory: planned volume does not fit in RAM"
    )


def test_create_app_requires_one_inference_source(service):
    with pytest.raises(ValueError, match="weights are required"):
        create_app(config=CONFIG)
    with pytest.raises(ValueError, match="cannot be provided together"):
        create_app("generator.pt", inference=service, config=CONFIG)
    with pytest.raises(ValueError, match="cannot be provided together"):
        create_app(inference=service, lr_weights="generator.pt", config=CONFIG)


@pytest.mark.parametrize(
    "exception",
    [
        MemoryError("allocation failed"),
        torch.OutOfMemoryError("allocation failed"),
        RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"),
    ],
)
def test_cpu_cuda_and_apple_gpu_oom_return_413(client, service, monkeypatch, exception):
    def fail(**kwargs):
        raise exception

    monkeypatch.setattr(service, "generate", fail)
    assert client.post("/api/generate", json={}).status_code == 413


@pytest.mark.parametrize(
    "payload",
    [
        {"shape": [100000, 100000, 100000]},
        {"shape": 1024},
        {"size": 1025},
        {"blocks": 65},
        {"blocks": [64, 64, 64]},
        {"shape": [-1, 32, 32]},
        {"shape": True},
        {"anchors": [{"image": [[0]], "axis": 0, "index": 0}] * 33},
        {"anchors": [{"image": [[0] * 1025], "axis": 0, "index": 0}]},
        {"vf_profile": {"axis": "z", "interpolation": "linear", "points": [[0, [1]]]}},
        {
            "vf_profile": {
                "axis": "z",
                "interpolation": "linear",
                "points": [[i / 256, [1.0]] for i in range(257)],
            }
        },
        {
            "vf_profile": {
                "axis": "x",
                "interpolation": "linear",
                "points": [[0, [1]], [1, [1]]],
            }
        },
    ],
)
def test_request_limits_reject_before_inference(client, service, payload):
    assert client.post("/api/generate", json=payload).status_code == 422
    assert not service.calls


def test_resolved_blocks_are_bounded_before_inference(client, service):
    # 64 LR blocks along z make 2 * (16 + 63 * 8) = 1040 HR voxels.
    assert client.post("/api/generate", json={"blocks": [64, 1, 1]}).status_code == 422
    assert not service.calls


def test_a_large_overlap_cannot_expand_a_request_into_millions_of_tiles(
    client, service
):
    response = client.post("/api/generate", json={"shape": 256, "overlap": 15})
    assert response.status_code == 422
    assert "LR tiles exceed" in response.json()["detail"]
    assert not service.calls


def test_memory_preflight_rejects_before_inference(client, service, monkeypatch):
    def reject(*args, **kwargs):
        raise MemoryError("preflight budget exceeded")

    monkeypatch.setattr(service, "check_memory", reject)
    response = client.post("/api/generate", json={})
    assert response.status_code == 413
    assert "preflight" in response.json()["detail"]
    assert not service.calls


def test_preflight_counts_anchors_and_guidance(client, service, monkeypatch):
    seen = []
    monkeypatch.setattr(
        service, "check_memory", lambda estimate, **_: seen.append(estimate)
    )
    anchor = {"image": [[0, 1], [1, 0]], "axis": 0, "index": 0}
    client.post("/api/generate", json={"guidance": 1.0, "format": "raw"})
    client.post(
        "/api/generate",
        json={"guidance": 2.0, "anchors": [anchor], "format": "raw"},
    )
    plain, guided = (estimate.low for estimate in seen)
    assert (
        guided.host_bytes + guided.device_bytes > plain.host_bytes + plain.device_bytes
    )


def test_an_oversized_body_is_rejected_before_json_decoding(service):
    config = CONFIG.model_copy(update={"max_request_bytes": 32})
    client = TestClient(create_app(inference=service, config=config))
    assert client.post("/api/generate", content=b" " * 33).status_code == 413
    assert not service.calls


@pytest.mark.parametrize("format", ["raw", "tiff"])
def test_slow_downloads_bound_retained_results_without_holding_generation(
    service, format
):
    app = create_app(inference=service, config=CONFIG)
    generate = next(
        route.endpoint for route in app.routes if route.path == "/api/generate"
    )
    scope = {"type": "http", "method": "POST", "headers": []}

    async def exercise():
        first_body = anyio.Event()
        finish_first = anyio.Event()
        first_done = anyio.Event()

        async def receive():
            await anyio.sleep_forever()

        async def slow_send(message):
            if message["type"] == "http.response.body":
                first_body.set()
                await finish_first.wait()

        async def send(message):
            pass

        first = generate(GenerateRequest(format=format))

        async def download_first():
            await first(scope, receive, slow_send)
            first_done.set()

        async with anyio.create_task_group() as group:
            group.start_soon(download_first)
            with anyio.fail_after(60):
                await first_body.wait()
                assert not app.state.generate_lock.locked()
                second = generate(GenerateRequest(format=format))
                assert len(service.calls) == 2
                with pytest.raises(HTTPException) as error:
                    generate(GenerateRequest(format=format))
                assert error.value.status_code == 503
                assert error.value.headers["Retry-After"] == "1"
                assert len(service.calls) == 2
                finish_first.set()
                await first_done.wait()
                third = generate(GenerateRequest(format=format))
                await second(scope, receive, send)
                await third(scope, receive, send)
        assert not app.state.generate_lock.locked()
        assert app.state.download_slots.acquire(blocking=False)
        assert app.state.download_slots.acquire(blocking=False)
        assert not app.state.download_slots.acquire(blocking=False)

    anyio.run(exercise)


@pytest.mark.parametrize(
    "failure", [MemoryError("oom"), ValueError("bad input"), OSError("disk")]
)
def test_failed_generation_returns_the_download_slot(service, monkeypatch, failure):
    cfg = CONFIG.model_copy(update={"max_inflight_downloads": 1})
    app = create_app(inference=service, config=cfg)
    client = TestClient(app)

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(service, "generate", fail)
    for _ in range(3):
        assert client.post("/api/generate", json={}).status_code in (413, 422, 500)
    assert app.state.download_slots.acquire(blocking=False)
    assert not app.state.download_slots.acquire(blocking=False)


@pytest.mark.parametrize(
    ("settings", "payload"),
    [
        ({"max_size": 3}, {}),
        ({"max_voxels": 63}, {}),
        ({"max_blocks": 1}, {"blocks": [2, 1, 1]}),
        ({"max_tiles": 1}, {"blocks": [3, 1, 1]}),
        (
            {"max_anchors": 1},
            {"anchors": [{"image": [[0]], "axis": 0, "index": 0}] * 2},
        ),
    ],
)
def test_configured_limits_reject_before_generation(service, settings, payload):
    cfg = CONFIG.model_copy(update=settings)
    with TestClient(create_app(inference=service, config=cfg)) as client:
        assert client.post("/api/generate", json=payload).status_code == 422
    assert not service.calls


@pytest.mark.parametrize(
    ("settings", "payload", "detail"),
    [
        ({"max_voxels": 63}, {}, "output shape [32, 32, 32] exceeds"),
        # Three LR blocks (32 voxels) with margin 8 fit in two 32-voxel tiles.
        ({"max_tiles": 1}, {"blocks": [3, 1, 1]}, "2 LR tiles exceed"),
    ],
)
def test_resolved_limits_report_the_request_and_the_limit(
    service, settings, payload, detail
):
    cfg = CONFIG.model_copy(update=settings)
    with TestClient(create_app(inference=service, config=cfg)) as client:
        message = client.post("/api/generate", json=payload).json()["detail"]
    assert detail in message
    assert str(next(iter(settings.values()))) in message


def test_health_reports_settings_the_model_cannot_tile_with(
    service, pipeline, monkeypatch
):
    # config/gen.yaml's own overlap 8 is half the tiny 16-voxel patch.
    settings = dataclasses.replace(pipeline.lr.settings, overlap=8)
    monkeypatch.setattr(pipeline.lr, "settings", settings)
    with TestClient(create_app(inference=service, config=CONFIG)) as client:
        response = client.get("/api/health")
    assert response.status_code == 500
    assert "2 * overlap < patch" in response.json()["detail"]


def test_prepare_uses_the_configured_image_limit(service):
    cfg = CONFIG.model_copy(update={"max_size": 2})
    with TestClient(create_app(inference=service, config=cfg)) as client:
        response = client.post("/api/prepare", json={"image": [[0, 1, 0]]})
    assert response.status_code == 422
    assert "image dimensions" in response.json()["detail"]


def test_apps_keep_independent_download_limits(service):
    cfg = CONFIG.model_copy(update={"max_inflight_downloads": 1})
    first = create_app(inference=service, config=cfg)
    second = create_app(inference=service, config=CONFIG)
    assert first.state.download_slots.acquire(blocking=False)
    assert not first.state.download_slots.acquire(blocking=False)
    assert second.state.download_slots.acquire(blocking=False)
    assert second.state.download_slots.acquire(blocking=False)
    assert not second.state.download_slots.acquire(blocking=False)
