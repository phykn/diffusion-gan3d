import errno
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import tifffile
import torch
from fastapi.testclient import TestClient

from src.evaluate.volume import VolumeMetrics
from src.predict.memory import estimate_memory, select_storage
from src.serve import app as server_module
from src.serve import response as response_module
from src.serve.app import create_app


class FakeInference:
    device = torch.device("cpu")
    crop_size = 96
    input_size = 128
    num_phases = 2

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def estimate_memory(self, *, blocks=None, shape=None, size=None, overlap=None):
        if blocks is not None and shape is not None:
            raise ValueError("blocks and shape cannot be provided together.")
        if blocks is not None:
            counts = (blocks,) * 3 if isinstance(blocks, int) else blocks
            stride = self.input_size - 2 * (8 if overlap is None else overlap)
            shape = tuple(self.input_size + (count - 1) * stride for count in counts)
        return estimate_memory(
            shape or size or self.input_size,
            self.num_phases,
            tile_size=self.input_size,
            overlap=8 if overlap is None else overlap,
        )

    def check_memory(self, estimate, **kwargs):
        return select_storage(
            kwargs["storage"],
            estimate,
            SimpleNamespace(device=self.device, model=None, num_phases=self.num_phases),
        )

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        if kwargs["blocks"] is not None and kwargs["shape"] is not None:
            raise ValueError("blocks and shape cannot be provided together.")
        return torch.arange(64, dtype=torch.uint8).reshape(4, 4, 4) % 2


@pytest.fixture
def service() -> FakeInference:
    return FakeInference()


@pytest.fixture
def client(
    service: FakeInference,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    front = tmp_path / "dist"
    assets = front / "assets"
    assets.mkdir(parents=True)
    (front / "index.html").write_text(
        '<div id="app"></div><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("", encoding="utf-8")
    monkeypatch.setattr(server_module, "FRONT_DIR", front)
    monkeypatch.setattr(
        server_module,
        "measure_volume",
        lambda _volume, device: VolumeMetrics(porosity=0.25, tortuosity=1.5),
    )
    return TestClient(create_app(inference=service))


def test_health_reports_loaded_device(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "device": "cpu",
        "crop_size": 96,
        "input_size": 128,
        "num_phases": 2,
    }


def test_frontend_is_served_by_the_api_process(client: TestClient) -> None:
    page = client.get("/")

    assert page.status_code == 200
    assert '<div id="app"></div>' in page.text
    assert "/assets/" in page.text


def test_vue_frontend_uses_fixed_boundary_anchor() -> None:
    app = Path("front/src/App.vue").read_text(encoding="utf-8")

    assert "anchors: [{ image, axis: 0, index: 0 }]" in app
    assert "include_metrics: true" in app
    assert ":crop-size=" in app
    assert ":input-size=" in app


def test_generate_returns_tiff_volume(
    client: TestClient,
    service: FakeInference,
) -> None:
    response = client.post(
        "/generate", json={"domain": 0, "seed": 3, "include_metrics": True}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert response.headers["x-volume-shape"] == "4,4,4"
    assert response.headers["x-porosity"] == "0.25"
    assert response.headers["x-tortuosity"] == "1.5"
    assert response.headers["x-tortuosity-axis"] == "1"
    volume = tifffile.imread(BytesIO(response.content))
    assert volume.shape == (4, 4, 4)
    assert volume.dtype == torch.empty((), dtype=torch.uint8).numpy().dtype
    assert service.calls[0]["seed"] == 3
    assert service.calls[0]["anchors"] == ()


def test_generate_can_return_raw_labels(client: TestClient) -> None:
    response = client.post("/generate", json={"format": "raw"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert len(response.content) == 4 * 4 * 4


def test_metrics_are_opt_in_and_remain_under_generation_lock(client, monkeypatch):
    calls = []

    def measure(volume, device):
        assert client.app.state.generate_lock.locked()
        calls.append(volume.shape)
        return VolumeMetrics(porosity=0.25, tortuosity=None)

    monkeypatch.setattr(server_module, "measure_volume", measure)
    response = client.post("/generate", json={"format": "raw"})
    assert response.status_code == 200 and not calls
    assert "x-porosity" not in response.headers
    response = client.post("/generate", json={"format": "raw", "include_metrics": True})
    assert len(calls) == 1 and response.headers["x-tortuosity"] == "unavailable"
    assert not client.app.state.generate_lock.locked()


def test_busy_request_is_rejected_without_waiting_or_releasing_owner_lock(
    client, service
):
    lock = client.app.state.generate_lock
    lock.acquire()
    try:
        response = client.post("/generate", json={"format": "raw"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert lock.locked()
        assert not service.calls
    finally:
        lock.release()
    assert client.post("/generate", json={"format": "raw"}).status_code == 200


@pytest.mark.parametrize(
    "failure,status",
    [
        (MemoryError("serialize allocation"), 413),
        (OSError(errno.ENOSPC, "disk full"), 507),
        (OSError(errno.EACCES, "cannot write"), 500),
    ],
)
def test_response_preparation_errors_cleanup_file_and_release_lock(
    client, monkeypatch, tmp_path, failure, status
):
    import tempfile

    monkeypatch.setattr(
        response_module,
        "NamedTemporaryFile",
        lambda **kwargs: tempfile.NamedTemporaryFile(dir=tmp_path, **kwargs),
    )
    original = response_module.tifffile.imwrite

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(response_module.tifffile, "imwrite", fail)
    response = client.post("/generate", json={})
    assert response.status_code == status
    assert not list(tmp_path.glob("*.tiff"))
    assert not client.app.state.generate_lock.locked()
    monkeypatch.setattr(response_module.tifffile, "imwrite", original)
    assert client.post("/generate", json={}).status_code == 200


def test_tiff_response_removes_tempfile_after_success(client, monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr(
        response_module,
        "NamedTemporaryFile",
        lambda **kwargs: tempfile.NamedTemporaryFile(dir=tmp_path, **kwargs),
    )
    response = client.post("/generate", json={})
    assert response.status_code == 200
    assert tifffile.imread(BytesIO(response.content)).shape == (4, 4, 4)
    assert not list(tmp_path.glob("*.tiff"))


def test_generate_decodes_anchor_and_scale_request(
    client: TestClient,
    service: FakeInference,
) -> None:
    response = client.post(
        "/generate",
        json={
            "anchors": [
                {
                    "image": [[0, 1], [1, 0]],
                    "axis": 0,
                    "index": 0,
                }
            ],
            "blocks": [2, 2, 2],
            "anchor_strength": 0.8,
        },
    )

    assert response.status_code == 200
    call = service.calls[0]
    assert call["blocks"] == (2, 2, 2)
    assert call["anchors"][0].axis == 0
    assert torch.equal(
        call["anchors"][0].image,
        torch.tensor(((0, 1), (1, 0)), dtype=torch.uint8),
    )
    assert call["anchor_strength"] == 0.8


def test_generate_rejects_non_rectangular_anchor(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json={
            "anchors": [
                {
                    "image": [[0, 1], [1]],
                    "axis": 0,
                    "index": 0,
                }
            ]
        },
    )

    assert response.status_code == 422


def test_generate_reports_inference_validation(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json={"blocks": 2, "shape": 16},
    )

    assert response.status_code == 422
    assert "blocks and shape" in response.json()["detail"]


def test_generate_reports_planned_memory_limit(
    client: TestClient,
    service: FakeInference,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(**_kwargs) -> torch.Tensor:
        raise MemoryError("planned volume does not fit in RAM")

    monkeypatch.setattr(service, "generate", reject)

    response = client.post("/generate", json={"blocks": 4})

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "generation exceeds available memory: planned volume does not fit in RAM"
    )


def test_create_app_requires_one_inference_source(service: FakeInference) -> None:
    with pytest.raises(ValueError, match="weights are required"):
        create_app()
    with pytest.raises(ValueError, match="cannot be provided together"):
        create_app("generator.pt", inference=service)


@pytest.mark.parametrize("exception", [MemoryError, torch.OutOfMemoryError])
def test_cpu_and_cuda_oom_return_413(client, service, monkeypatch, exception):
    def fail(**kwargs):
        raise exception("allocation failed")

    monkeypatch.setattr(service, "generate", fail)
    assert client.post("/generate", json={}).status_code == 413


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
    ],
)
def test_request_limits_reject_before_inference(client, service, payload):
    assert client.post("/generate", json=payload).status_code == 422
    assert not service.calls


def test_resolved_blocks_are_bounded_before_inference(client, service):
    assert client.post("/generate", json={"blocks": [10, 1, 1]}).status_code == 422
    assert not service.calls


def test_extreme_overlap_cannot_expand_a_small_request_into_millions_of_tiles(
    client, service
):
    response = client.post("/generate", json={"shape": 512, "overlap": 63})
    assert response.status_code == 422
    assert "tile count" in response.json()["detail"]
    assert not service.calls


def test_memory_preflight_rejects_before_inference(client, service, monkeypatch):
    def reject(*args, **kwargs):
        raise MemoryError("preflight budget exceeded")

    monkeypatch.setattr(service, "check_memory", reject)
    response = client.post("/generate", json={})
    assert response.status_code == 413
    assert "preflight" in response.json()["detail"]
    assert not service.calls


def test_oversized_body_is_rejected_before_json_decoding(service, monkeypatch):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BYTES", 32)
    client = TestClient(create_app(inference=service))
    assert client.post("/generate", content=b" " * 33).status_code == 413
    assert not service.calls
