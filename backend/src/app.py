import errno
import math
from pathlib import Path
from threading import BoundedSemaphore, Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.src.config import ServerConfig, load_config
from backend.src.middleware import RequestSizeLimit
from backend.src.response import DownloadResponse, volume_response
from backend.src.schema import GenerateRequest, PrepareRequest
from src.evaluate.volume import measure_volume
from src.predict.inference import InferenceAPI

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(
    weights: str | Path | None = None,
    device: str | torch.device | None = None,
    inference: InferenceAPI | None = None,
    *,
    config: ServerConfig | None = None,
    max_inflight_downloads: int | None = None,
) -> FastAPI:
    config = load_config() if config is None else config
    if max_inflight_downloads is not None:
        config = ServerConfig.model_validate(
            {**config.model_dump(), "max_inflight_downloads": max_inflight_downloads}
        )
    if inference is None:
        if weights is None:
            raise ValueError("weights are required when inference is not provided.")
        inference = InferenceAPI(weights, device=device)
    elif weights is not None:
        raise ValueError("weights and inference cannot be provided together.")

    app = FastAPI(
        title="Diffusion-GAN 3D inference",
    )
    app.add_middleware(RequestSizeLimit, limit=config.max_request_bytes)
    app.state.config = config
    app.state.inference = inference
    app.state.generate_lock = Lock()
    app.state.download_slots = BoundedSemaphore(config.max_inflight_downloads)

    @app.get("/health")
    def health() -> dict:
        generator = app.state.inference.generator
        return {
            "status": "ready",
            "device": str(app.state.inference.device),
            "crop_size": app.state.inference.crop_size,
            "input_size": app.state.inference.input_size,
            "num_phases": app.state.inference.num_phases,
            "num_domains": generator.num_domains,
            "height_enabled": generator.height_data is not None,
            "height_extents": (generator.height_data or {}).get("height_extents", {}),
        }

    @app.post("/prepare")
    def prepare(request: PrepareRequest) -> dict:
        try:
            request = request.check_limits(config)
            image = app.state.inference.prepare_image(
                torch.tensor(request.image, dtype=torch.long)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"image": image.tolist()}

    @app.post("/generate", response_class=StreamingResponse)
    def generate(request: GenerateRequest) -> Response:
        try:
            request = request.check_limits(config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
            if (
                max(estimate.shape) > config.max_size
                or math.prod(estimate.shape) > config.max_voxels
            ):
                raise ValueError("resolved output shape exceeds the server size limit")
            if estimate.tile_count > config.max_total_blocks:
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
                height_extent=request.height_extent,
                vf_profile=request.vf_profile,
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

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app
