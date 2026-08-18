from io import BytesIO
from pathlib import Path

import pytest
import tifffile
import torch
from fastapi.testclient import TestClient

from src.api import server as server_module
from src.api.metrics import VolumeMetrics
from src.api.server import create_app


class FakeInference:
    device = torch.device("cpu")
    crop_size = 96
    input_size = 128
    num_phases = 2

    def __init__(self) -> None:
        self.calls: list[dict] = []

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
    assert ":crop-size=" in app
    assert ":input-size=" in app


def test_generate_returns_tiff_volume(
    client: TestClient,
    service: FakeInference,
) -> None:
    response = client.post("/generate", json={"domain": 0, "seed": 3})

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
