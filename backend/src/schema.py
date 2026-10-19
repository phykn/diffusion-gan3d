import math
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field, field_validator

from backend.src.config import ServerConfig
from src.anchor import PlaneAnchor

Dimension = Annotated[int, Field(ge=1, strict=True)]
Index = Annotated[int, Field(ge=0, strict=True)]
PhaseID = Annotated[int, Field(ge=0, le=255, strict=True)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
LabelRow = Annotated[list[PhaseID], Field(min_length=1)]
LabelImage = Annotated[list[LabelRow], Field(min_length=1)]
ProbabilityRow = Annotated[list[Probability], Field(min_length=1)]
ProbabilityPlane = Annotated[list[ProbabilityRow], Field(min_length=1)]
ProbabilityImage = Annotated[
    list[ProbabilityPlane], Field(min_length=1, max_length=256)
]


class AnchorRequest(BaseModel):
    image: LabelImage | ProbabilityImage
    axis: Annotated[int, Field(ge=0, le=2, strict=True)]
    index: Index
    position: tuple[Index, Index] | None = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, image):
        return check_rectangular(image)

    def check_limits(self, cfg: ServerConfig) -> None:
        check_image_size(self.image, cfg.max_size)
        if self.index >= cfg.max_size or (
            self.position is not None and max(self.position) >= cfg.max_size
        ):
            raise ValueError("anchor coordinates exceed the server size limit")

    def to_anchor(self) -> PlaneAnchor:
        return PlaneAnchor(
            image=torch.tensor(self.image),
            axis=self.axis,
            index=self.index,
            position=self.position,
        )


class GenerateRequest(BaseModel):
    anchors: list[AnchorRequest] = Field(default_factory=list)
    blocks: Dimension | tuple[Dimension, Dimension, Dimension] | None = None
    shape: Dimension | tuple[Dimension, Dimension, Dimension] | None = None
    size: Dimension | None = None
    vf: list[Probability] | None = Field(default=None, max_length=256)
    domain: Index | None = None
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1, strict=True)
    guidance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    anchor_strength: Probability | None = None
    height_origin: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    height_extent: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    vf_profile: dict | None = None
    overlap: Index | None = None
    storage: Literal["auto", "cpu", "cuda"] = "auto"
    progress: bool = False
    format: Literal["tiff", "raw"] = "tiff"
    include_metrics: bool = False

    def check_limits(self, cfg: ServerConfig) -> None:
        if len(self.anchors) > cfg.max_anchors:
            raise ValueError("anchor count exceeds the server limit")
        for anchor in self.anchors:
            anchor.check_limits(cfg)
        if self.overlap is not None and self.overlap > cfg.max_size // 2:
            raise ValueError("overlap exceeds the server size limit")
        for value, dimension, maximum, label in (
            (self.shape, cfg.max_size, cfg.max_voxels, "shape voxels"),
            (self.size, cfg.max_size, cfg.max_voxels, "size voxels"),
            (self.blocks, cfg.max_blocks, cfg.max_total_blocks, "block count"),
        ):
            if value is None:
                continue
            shape = (value,) * 3 if isinstance(value, int) else value
            if max(shape) > dimension or math.prod(shape) > maximum:
                raise ValueError(f"{label} exceeds the server limit")


class PrepareRequest(BaseModel):
    image: LabelImage

    @field_validator("image")
    @classmethod
    def validate_image(cls, image):
        return check_rectangular(image)

    def check_limits(self, cfg: ServerConfig) -> None:
        check_image_size(self.image, cfg.max_size)


def image_planes(image) -> list:
    return image if isinstance(image[0][0], list) else [image]


def check_rectangular(image):
    planes = image_planes(image)
    height, width = len(planes[0]), len(planes[0][0])
    if any(
        len(plane) != height or any(len(row) != width for row in plane)
        for plane in planes
    ):
        raise ValueError("image rows must have equal length")
    return image


def check_image_size(image, maximum: int) -> None:
    if any(
        len(plane) > maximum or any(len(row) > maximum for row in plane)
        for plane in image_planes(image)
    ):
        raise ValueError("image dimensions exceed the server size limit")
