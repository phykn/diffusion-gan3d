import errno
import math
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from src.anchor import PlaneAnchor
from src.evaluate.volume import measure_volume
from src.predict.inference import InferenceAPI
from src.serve.response import DownloadResponse, volume_response

FRONT_DIR = Path(__file__).resolve().parents[2] / "front" / "dist"
MAX_SIZE = 1024
MAX_VOXELS = 512**3
MAX_BLOCKS = 64
MAX_TOTAL_BLOCKS = 4096
MAX_ANCHORS = 32
MAX_PHASES = 256
MAX_REQUEST_BYTES = 16 * 1024**2
Dimension = Annotated[int, Field(ge=1, le=MAX_SIZE, strict=True)]
BlockCount = Annotated[int, Field(ge=1, le=MAX_BLOCKS, strict=True)]
PhaseID = Annotated[int, Field(ge=0, le=255, strict=True)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
LabelRow = Annotated[list[PhaseID], Field(min_length=1, max_length=MAX_SIZE)]
LabelImage = Annotated[list[LabelRow], Field(min_length=1, max_length=MAX_SIZE)]
ProbabilityRow = Annotated[list[Probability], Field(min_length=1, max_length=MAX_SIZE)]
ProbabilityPlane = Annotated[
    list[ProbabilityRow], Field(min_length=1, max_length=MAX_SIZE)
]
ProbabilityImage = Annotated[
    list[ProbabilityPlane], Field(min_length=1, max_length=MAX_PHASES)
]


class RequestSizeLimit:
    """Bound JSON bodies before parsing, including chunked uploads."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                response = JSONResponse(
                    status_code=413, content={"detail": "request body is too large"}
                )
                return await response(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        del chunks

        async def bounded_receive():
            nonlocal body
            if body is not None:
                result, body = body, None
                return {"type": "http.request", "body": result, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


class AnchorRequest(BaseModel):
    image: LabelImage | ProbabilityImage
    axis: Annotated[int, Field(ge=0, le=2)]
    index: Annotated[int, Field(ge=0, lt=MAX_SIZE)]
    position: (
        tuple[
            Annotated[int, Field(ge=0, lt=MAX_SIZE)],
            Annotated[int, Field(ge=0, lt=MAX_SIZE)],
        ]
        | None
    ) = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, image):
        if not image or not image[0]:
            raise ValueError("anchor image must not be empty")
        planes = image if isinstance(image[0][0], list) else [image]
        height, width = len(planes[0]), len(planes[0][0])
        if width == 0 or any(
            len(plane) != height or any(len(row) != width for row in plane)
            for plane in planes
        ):
            raise ValueError("anchor image rows must have equal length")
        return image

    def to_anchor(self) -> PlaneAnchor:
        return PlaneAnchor(
            image=torch.tensor(self.image),
            axis=self.axis,
            index=self.index,
            position=self.position,
        )


class GenerateRequest(BaseModel):
    anchors: list[AnchorRequest] = Field(default_factory=list, max_length=MAX_ANCHORS)
    blocks: BlockCount | tuple[BlockCount, BlockCount, BlockCount] | None = None
    shape: Dimension | tuple[Dimension, Dimension, Dimension] | None = None
    size: Dimension | None = None
    vf: list[Probability] | None = Field(default=None, max_length=MAX_PHASES)
    domain: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)
    guidance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    anchor_strength: Probability | None = None
    height_origin: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    overlap: int | None = Field(default=None, ge=0, le=MAX_SIZE // 2)
    storage: Literal["auto", "cpu", "cuda"] = "auto"
    progress: bool = False
    format: Literal["tiff", "raw"] = "tiff"
    include_metrics: bool = False

    @model_validator(mode="after")
    def validate_volume(self):
        for value, maximum, label in (
            (self.shape, MAX_VOXELS, "shape voxels"),
            (self.size, MAX_VOXELS, "size voxels"),
            (self.blocks, MAX_TOTAL_BLOCKS, "block count"),
        ):
            if value is not None:
                count = value**3 if isinstance(value, int) else math.prod(value)
                if count > maximum:
                    raise ValueError(f"{label} must not exceed {maximum}")
        return self


class PrepareRequest(BaseModel):
    image: LabelImage


def create_app(
    weights: str | Path | None = None,
    device: str | torch.device | None = None,
    inference: InferenceAPI | None = None,
    *,
    max_inflight_downloads: int = 2,
) -> FastAPI:
    if type(max_inflight_downloads) is not int or max_inflight_downloads < 1:
        raise ValueError("max_inflight_downloads must be a positive integer.")
    if inference is None:
        if weights is None:
            raise ValueError("weights are required when inference is not provided.")
        inference = InferenceAPI(weights, device=device)
    elif weights is not None:
        raise ValueError("weights and inference cannot be provided together.")

    app = FastAPI(
        title="Diffusion-GAN 3D inference",
    )
    app.add_middleware(RequestSizeLimit, limit=MAX_REQUEST_BYTES)
    app.state.inference = inference
    app.state.generate_lock = Lock()
    app.state.download_slots = BoundedSemaphore(max_inflight_downloads)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        return {
            "status": "ready",
            "device": str(app.state.inference.device),
            "crop_size": app.state.inference.crop_size,
            "input_size": app.state.inference.input_size,
            "num_phases": app.state.inference.num_phases,
        }

    @app.post("/prepare")
    def prepare(request: PrepareRequest) -> dict:
        try:
            image = app.state.inference.prepare_image(
                torch.tensor(request.image, dtype=torch.long)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"image": image.tolist()}

    @app.post("/generate", response_class=StreamingResponse)
    def generate(request: GenerateRequest) -> Response:
        if not app.state.download_slots.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="download capacity is full; retry later",
                headers={"Retry-After": "1"},
            )
        if not app.state.generate_lock.acquire(blocking=False):
            app.state.download_slots.release()
            raise HTTPException(
                status_code=503,
                detail="generation is busy; retry later",
                headers={"Retry-After": "1"},
            )
        response_owns_slot = False
        try:
            estimate = app.state.inference.estimate_memory(
                blocks=request.blocks,
                shape=request.shape,
                size=request.size,
                overlap=request.overlap,
            )
            if max(estimate.shape) > MAX_SIZE or math.prod(estimate.shape) > MAX_VOXELS:
                raise ValueError("resolved output shape exceeds the server size limit")
            if estimate.tile_count > MAX_TOTAL_BLOCKS:
                raise ValueError("resolved tile count exceeds the server block limit")
            app.state.inference.check_memory(
                estimate,
                storage=request.storage,
                tiled=request.blocks is not None or request.shape is not None,
            )
            volume = app.state.inference.generate(
                anchors=tuple(anchor.to_anchor() for anchor in request.anchors),
                blocks=request.blocks,
                shape=request.shape,
                size=request.size,
                vf=request.vf,
                domain=request.domain,
                seed=request.seed,
                guidance=request.guidance,
                anchor_strength=request.anchor_strength,
                height_origin=request.height_origin,
                overlap=request.overlap,
                storage=request.storage,
                progress=request.progress,
            )
            headers = {
                "X-Volume-Shape": ",".join(str(value) for value in volume.shape),
                "X-Volume-Dtype": "uint8",
            }
            if request.include_metrics:
                metrics = measure_volume(volume, device=app.state.inference.device)
                headers.update(
                    {
                        "X-Porosity": f"{metrics.porosity:.8g}",
                        "X-Tortuosity": "unavailable"
                        if metrics.tortuosity is None
                        else f"{metrics.tortuosity:.8g}",
                        "X-Pore-Phase": "0",
                        "X-Tortuosity-Axis": "1",
                    }
                )
            response = DownloadResponse(
                volume_response(volume, request.format, headers),
                app.state.download_slots.release,
            )
            response_owns_slot = True
            return response
        except (MemoryError, torch.OutOfMemoryError) as exc:
            raise HTTPException(
                status_code=413,
                detail=f"generation exceeds available memory: {exc}",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=507 if exc.errno == errno.ENOSPC else 500,
                detail=f"could not prepare volume response: {exc}",
            ) from exc
        finally:
            app.state.generate_lock.release()
            if not response_owns_slot:
                app.state.download_slots.release()

    if FRONT_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="front")
    return app
