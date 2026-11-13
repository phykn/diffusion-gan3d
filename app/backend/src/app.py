import math
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from app.backend.src.config import load_server_config
from app.backend.src.pipeline import HighResolutionPipeline
from app.backend.src.response import volume_response
from app.backend.src.schema import GenerateRequest, PrepareRequest
from src.axes import PORE_PHASE

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(
    weights=None, device=None, inference=None, *, lr_weights=None, config=None
):
    if inference is None:
        if weights is None:
            raise ValueError("weights are required when inference is not provided.")
        inference = HighResolutionPipeline(weights, lr_weights, device)
    elif weights is not None or lr_weights is not None:
        raise ValueError("weights and inference cannot be provided together.")
    config = load_server_config() if config is None else config

    app = FastAPI(title="Diffusion-GAN 3D inference")
    generate_lock = Lock()
    app.state.generate_lock = generate_lock

    @app.get("/api/health")
    def health():
        try:
            return describe_model(inference, config)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/prepare")
    def prepare(request: PrepareRequest):
        try:
            request.check_limits(config)
            labels = torch.tensor(request.image, dtype=torch.long)
            return {"image": inference.prepare_image(labels).tolist()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/generate", response_class=Response)
    def generate(request: GenerateRequest):
        if not generate_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="generation is busy; retry later",
                headers={"Retry-After": "1"},
            )
        try:
            request.check_limits(config)
            labels = generate_volume(inference, request, config)
            return volume_response(labels, inference.device, request.tortuosity)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (MemoryError, RuntimeError) as exc:
            # MPS reports allocator exhaustion as a plain RuntimeError.
            if not isinstance(
                exc, (MemoryError, torch.OutOfMemoryError)
            ) and "out of memory" not in str(exc):
                raise
            raise HTTPException(
                status_code=413, detail=f"generation exceeds available memory: {exc}"
            ) from exc
        finally:
            generate_lock.release()

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def describe_model(inference, config):
    limit = 1
    low, high = 2, config.max_blocks
    while low <= high:
        count = (low + high) // 2
        shape = inference.estimate_memory(blocks=(count, 1, 1)).shape
        if max(shape) <= config.max_size:
            limit = count
            low = count + 1
        else:
            high = count - 1
    return {
        "device": str(inference.device),
        "crop_size": inference.crop_size,
        "num_phases": inference.num_phases,
        "height_enabled": inference.height_enabled,
        "height_extent": inference.height_extent,
        "pore_phase": PORE_PHASE,
        "max_axis_blocks": limit,
    }


def generate_volume(inference, request, config):
    geometry = {
        "blocks": request.blocks,
        "shape": request.shape,
        "anchors": tuple(anchor.to_anchor() for anchor in request.anchors),
    }
    estimate = inference.estimate_memory(**geometry)
    if (
        max(estimate.shape) > config.max_size
        or math.prod(estimate.shape) > config.max_voxels
    ):
        raise ValueError(
            f"output shape {list(estimate.shape)} exceeds the server limits of "
            f"{config.max_size} voxels per axis and {config.max_voxels} voxels"
        )
    if estimate.tile_count > config.max_tiles:
        raise ValueError(
            f"{estimate.tile_count} LR tiles exceed the server limit of {config.max_tiles}"
        )
    inference.check_memory(estimate)
    return inference.generate(
        **geometry,
        seed=request.seed,
        height_origin=request.height_origin,
        height_extent=request.height_extent,
    )
