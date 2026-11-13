import math
from typing import Annotated

import torch
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from src.api import PlaneAnchor

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


def image_planes(image):
    if isinstance(image[0][0], list):
        return image
    return (image,)


def check_rectangular(image):
    planes = image_planes(image)
    rows, columns = len(planes[0]), len(planes[0][0])
    for plane in planes:
        if len(plane) != rows or any(len(row) != columns for row in plane):
            raise ValueError("image rows must have equal length")
    return image


def check_image_size(image, maximum):
    first = image_planes(image)[0]
    if max(len(first), len(first[0])) > maximum:
        raise ValueError("image dimensions exceed the server size limit")


def as_shape(value):
    return (value,) * 3 if isinstance(value, int) else value


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnchorRequest(Request):
    image: Annotated[LabelImage | ProbabilityImage, AfterValidator(check_rectangular)]
    axis: Annotated[int, Field(ge=0, le=2, strict=True)]
    index: Index
    position: tuple[Index, Index] | None = None

    def check_limits(self, cfg):
        check_image_size(self.image, cfg.max_size)
        if self.index >= cfg.max_size or (
            self.position is not None and max(self.position) >= cfg.max_size
        ):
            raise ValueError("anchor coordinates exceed the server size limit")

    def to_anchor(self):
        return PlaneAnchor(
            torch.tensor(self.image), self.axis, self.index, self.position
        )


class GenerateRequest(Request):
    anchors: list[AnchorRequest] = Field(default_factory=list, max_length=1)
    blocks: Dimension | tuple[Dimension, Dimension, Dimension] | None = None
    shape: Dimension | tuple[Dimension, Dimension, Dimension] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1, strict=True)
    height_origin: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    height_extent: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    tortuosity: bool = False

    def check_limits(self, cfg):
        for anchor in self.anchors:
            anchor.check_limits(cfg)
        # Reject huge requests before planning; app.py checks the resolved shape.
        if self.shape is not None:
            shape = as_shape(self.shape)
            if max(shape) > cfg.max_size or math.prod(shape) > cfg.max_voxels:
                raise ValueError("shape exceeds the server limit")
        if self.blocks is not None and max(as_shape(self.blocks)) > cfg.max_blocks:
            raise ValueError("block count exceeds the server limit")


class PrepareRequest(Request):
    image: Annotated[LabelImage, AfterValidator(check_rectangular)]

    def check_limits(self, cfg):
        check_image_size(self.image, cfg.max_size)
