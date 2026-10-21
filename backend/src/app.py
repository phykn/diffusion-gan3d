import errno
import math
from pathlib import Path
from threading import BoundedSemaphore, Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.src.config import ServerConfig, load_server_config
from backend.src.middleware import RequestSizeLimit
from backend.src.response import DownloadResponse, volume_response
from backend.src.schema import GenerateRequest, PrepareRequest
from src.api import LowResolutionAPI

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(
    weights: str | Path | None = None,
    device: str | torch.device | None = None,
    inference: LowResolutionAPI | None = None,
    *,
    config: ServerConfig | None = None,
) -> FastAPI:
    config = load_server_config() if config is None else config
    if inference is None:
        if weights is None:
            raise ValueError("weights are required when inference is not provided.")
        inference = LowResolutionAPI(weights, device=device)
    elif weights is not None:
        raise ValueError("weights and inference cannot be provided together.")

    app = FastAPI(
        title="Diffusion-GAN 3D inference",
    )
    app.add_middleware(RequestSizeLimit, limit=config.max_request_bytes)
    app.state.generate_lock = Lock()
    app.state.download_slots = BoundedSemaphore(config.max_inflight_downloads)

    @app.get("/api/health")
    def health() -> dict:
        generator = inference.generator
        return {
            "device": str(inference.device),
            "crop_size": inference.crop_size,
            "num_phases": inference.num_phases,
            "num_domains": generator.num_domains,
            "height_enabled": generator.height_data is not None,
            "height_extents": (generator.height_data or {}).get("height_extents", {}),
            "max_blocks": _axis_block_limit(inference, config),
        }

    @app.post("/api/prepare")
    def prepare(request: PrepareRequest) -> dict:
        try:
            request.check_limits(config)
            image = inference.prepare_image(
                torch.tensor(request.image, dtype=torch.long)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"image": image.tolist()}

    @app.post("/api/generate", response_class=StreamingResponse)
    def generate(request: GenerateRequest) -> Response:
        try:
            request.check_limits(config)
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
            try:
                volume = _generate_volume(inference, request, config)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                content = volume_response(
                    volume,
                    request.format,
                    device=inference.device,
                    tortuosity=request.tortuosity,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"could not prepare volume response: {exc}",
                ) from exc
            response = DownloadResponse(content, app.state.download_slots.release)
            response_owns_slot = True
            return response
        except (MemoryError, torch.OutOfMemoryError) as exc:
            raise HTTPException(
                status_code=413,
                detail=f"generation exceeds available memory: {exc}",
            ) from exc
        except RuntimeError as exc:
            # MPS reports exhaustion as a plain RuntimeError.
            if "out of memory" not in str(exc):
                raise
            raise HTTPException(
                status_code=413,
                detail=f"generation exceeds available memory: {exc}",
            ) from exc
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


def _axis_block_limit(inference: LowResolutionAPI, config: ServerConfig) -> int:
    """Largest block count on one axis whose default tiling fits max_size."""
    limit = 1
    for count in range(2, config.max_blocks + 1):
        if max(inference.estimate_memory(blocks=(count, 1, 1)).shape) > config.max_size:
            break
        limit = count
    return limit


def _generate_volume(
    inference: LowResolutionAPI, request: GenerateRequest, config: ServerConfig
) -> torch.Tensor:
    estimate = inference.estimate_memory(
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
    if estimate.tile_count > config.max_tiles:
        raise ValueError("resolved tile count exceeds the server tile limit")
    inference.check_memory(estimate, storage=request.storage)
    return inference.generate(
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
    )
