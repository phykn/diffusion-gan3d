"""HTTP API on the tiny SR run."""

import dataclasses
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.backend.src import app as server_module
from app.backend.src import response as response_module
from app.backend.src.app import create_app
from app.backend.src.config import load_server_config
from app.backend.src.pipeline import HighResolutionPipeline

CONFIG = load_server_config(
    Path(__file__).resolve().parents[1] / "fixtures/config/backend.yaml"
)


SIZE_OPTIONS = ("blocks", "shape", "anchors")


class Recorder:
    """A pipeline that records what each request asks it to generate.

    Unless real is set, it returns zeros of the planned shape: most tests
    check the HTTP layer, not the volume.
    """

    def __init__(self, pipeline, real=False):
        self.pipeline = pipeline
        self.real = real
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.real:
            return self.pipeline.generate(**kwargs)
        options = {key: kwargs[key] for key in SIZE_OPTIONS if key in kwargs}
        shape = self.pipeline.estimate_memory(**options).shape
        return torch.zeros(shape, dtype=torch.uint8)


def tiny_pipeline(run):
    pipeline = HighResolutionPipeline(run, device="cpu")
    # config/gen.yaml's overlap 8 cannot tile the tiny 16-voxel LR patch.
    pipeline.lr.settings = dataclasses.replace(pipeline.lr.settings, overlap=4)
    return pipeline


@pytest.fixture(scope="session")
def pipeline(tiny_sr_run):
    return tiny_pipeline(tiny_sr_run)


@pytest.fixture
def service(pipeline):
    return Recorder(pipeline)


@pytest.fixture
def client(service, tmp_path, monkeypatch):
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


def test_generate_returns_raw_labels_and_metadata(client, service):
    response = client.post("/api/generate", json={"seed": 3, "tortuosity": True})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-volume-shape"] == "32,32,32"
    assert response.headers["x-tortuosity"] == "1.5"
    assert len(response.content) == 32**3
    assert service.calls[0]["seed"] == 3
    assert service.calls[0]["anchors"] == ()


def test_the_same_seed_gives_the_same_volume(client, service):
    service.real = True
    first = client.post("/api/generate", json={"seed": 5})
    second = client.post("/api/generate", json={"seed": 5})
    assert first.content == second.content


def test_generate_forwards_height(tiny_anode_sr_run):
    service = Recorder(tiny_pipeline(tiny_anode_sr_run), real=True)
    client = TestClient(create_app(inference=service, config=CONFIG))
    response = client.post(
        "/api/generate", json={"height_origin": 16, "height_extent": 64}
    )
    assert response.status_code == 200
    assert service.calls[0]["height_origin"] == 16
    assert service.calls[0]["height_extent"] == 64
    response = client.post("/api/generate", json={"height_origin": 48})
    assert response.status_code == 422
    assert "outside the source image height" in response.json()["detail"]


def test_raw_blocks_match_the_volume_shape_header(client):
    response = client.post("/api/generate", json={"blocks": [1, 2, 2]})
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
    response = client.post("/api/generate", json={"seed": 2**64 - 1})
    assert response.status_code == 200
    assert service.calls[0]["seed"] == 2**64 - 1


def test_tortuosity_is_opt_in_and_stays_under_the_generation_lock(client, monkeypatch):
    calls = []

    def measure(volume, device):
        assert client.app.state.generate_lock.locked()
        calls.append(volume.shape)
        return None

    monkeypatch.setattr(response_module, "pore_tortuosity", measure)
    response = client.post("/api/generate", json={})
    assert response.status_code == 200 and not calls
    assert "x-tortuosity" not in response.headers
    response = client.post("/api/generate", json={"tortuosity": True})
    assert len(calls) == 1 and response.headers["x-tortuosity"] == "unavailable"
    assert not client.app.state.generate_lock.locked()


def test_a_busy_request_is_rejected_without_waiting(client, service):
    lock = client.app.state.generate_lock
    lock.acquire()
    try:
        response = client.post("/api/generate", json={})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert lock.locked()
        assert not service.calls
    finally:
        lock.release()
    assert client.post("/api/generate", json={}).status_code == 200


def test_a_centered_anchor_and_blocks_reach_the_generator(client, service):
    response = client.post(
        "/api/generate",
        json={
            "anchors": [
                {"image": [[0, 1], [1, 0]], "axis": 0, "index": 0, "position": None}
            ],
            "blocks": [2, 2, 2],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-volume-shape"] == "48,48,48"
    call = service.calls[0]
    assert call["blocks"] == (2, 2, 2)
    anchor = call["anchors"][0]
    assert (anchor.axis, anchor.index, anchor.position) == (0, 0, None)
    assert anchor.image.tolist() == [[0, 1], [1, 0]]


def test_a_prepared_crop_anchors_the_first_section(client, service, pipeline):
    service.real = True
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
    assert "only one of blocks and shape" in response.json()["detail"]


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
        {"anchors": [{"image": [[0]], "axis": 0, "index": 0}] * 2},
        {"anchors": [{"image": [[0] * 1025], "axis": 0, "index": 0}]},
        {"vf_profile": {}},
        {"format": "tiff"},
        {"guidance": 2.0},
    ],
)
def test_request_limits_reject_before_inference(client, service, payload):
    assert client.post("/api/generate", json=payload).status_code == 422
    assert not service.calls


def test_resolved_blocks_are_bounded_before_inference(client, service):
    # 64 LR blocks along z make 2 * (16 + 63 * 8) = 1040 HR voxels.
    assert client.post("/api/generate", json={"blocks": [64, 1, 1]}).status_code == 422
    assert not service.calls


def test_memory_preflight_rejects_before_inference(client, service, monkeypatch):
    def reject(*args, **kwargs):
        raise MemoryError("preflight budget exceeded")

    monkeypatch.setattr(service, "check_memory", reject)
    response = client.post("/api/generate", json={})
    assert response.status_code == 413
    assert "preflight" in response.json()["detail"]
    assert not service.calls


def test_preflight_counts_anchors(client, service, monkeypatch):
    seen = []
    monkeypatch.setattr(
        service, "check_memory", lambda estimate, **_: seen.append(estimate)
    )
    anchor = {"image": [[0, 1], [1, 0]], "axis": 0, "index": 0}
    client.post("/api/generate", json={})
    client.post(
        "/api/generate",
        json={"anchors": [anchor]},
    )
    plain, guided = (estimate.low for estimate in seen)
    assert (
        guided.host_bytes + guided.device_bytes > plain.host_bytes + plain.device_bytes
    )


@pytest.mark.parametrize(
    ("settings", "payload"),
    [
        ({"max_size": 3}, {}),
        ({"max_voxels": 63}, {}),
        ({"max_blocks": 1}, {"blocks": [2, 1, 1]}),
        ({"max_tiles": 1}, {"blocks": [3, 1, 1]}),
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
