from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import tifffile
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.anchor import PlaneAnchor
from src.evaluate.volume import measure_volume
from src.predict.inference import InferenceAPI

FRONT_DIR = Path(__file__).resolve().parents[2] / "front" / "dist"


class AnchorRequest(BaseModel):
    image: list[list[int]] | list[list[list[float]]]
    axis: Annotated[int, Field(ge=0, le=2)]
    index: Annotated[int, Field(ge=0)]
    position: tuple[int, int] | None = None

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
    anchors: list[AnchorRequest] = Field(default_factory=list)
    blocks: int | tuple[int, int, int] | None = None
    shape: int | tuple[int, int, int] | None = None
    size: int | None = Field(default=None, ge=1)
    vf: list[float] | None = None
    domain: int | None = None
    seed: int | None = Field(default=None, ge=0)
    guidance: float | None = None
    anchor_strength: float | None = None
    height_origin: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    overlap: int | None = Field(default=None, ge=0)
    storage: Literal["auto", "cpu", "cuda"] = "auto"
    progress: bool = False
    format: Literal["tiff", "raw"] = "tiff"


class PrepareRequest(BaseModel):
    image: list[list[int]]


def create_app(
    weights: str | Path | None = None,
    device: str | torch.device | None = None,
    inference: InferenceAPI | None = None,
) -> FastAPI:
    if inference is None:
        if weights is None:
            raise ValueError("weights are required when inference is not provided.")
        inference = InferenceAPI(weights, device=device)
    elif weights is not None:
        raise ValueError("weights and inference cannot be provided together.")

    app = FastAPI(
        title="Diffusion-GAN 3D inference",
        version="2",
    )
    app.state.inference = inference
    app.state.generate_lock = Lock()

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
    def generate(request: GenerateRequest) -> StreamingResponse:
        try:
            with app.state.generate_lock:
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
                metrics = measure_volume(
                    volume,
                    device=app.state.inference.device,
                )
        except MemoryError as exc:
            raise HTTPException(
                status_code=413,
                detail=f"generation exceeds available memory: {exc}",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        tortuosity = (
            "unavailable" if metrics.tortuosity is None else f"{metrics.tortuosity:.8g}"
        )
        headers = {
            "X-Volume-Shape": ",".join(str(value) for value in volume.shape),
            "X-Volume-Dtype": "uint8",
            "X-Porosity": f"{metrics.porosity:.8g}",
            "X-Tortuosity": tortuosity,
            "X-Pore-Phase": "0",
            "X-Tortuosity-Axis": "1",
        }
        if request.format == "raw":
            output = BytesIO(volume.numpy().tobytes(order="C"))
            media_type = "application/octet-stream"
        else:
            output = BytesIO()
            tifffile.imwrite(output, volume.numpy())
            output.seek(0)
            media_type = "image/tiff"
            headers["Content-Disposition"] = 'attachment; filename="volume.tiff"'
        return StreamingResponse(
            output,
            media_type=media_type,
            headers=headers,
        )

    if FRONT_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="front")
    return app
