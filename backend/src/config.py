from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_size: PositiveInt
    max_voxels: PositiveInt
    max_blocks: PositiveInt
    max_total_blocks: PositiveInt
    max_anchors: PositiveInt
    max_request_bytes: PositiveInt
    max_inflight_downloads: PositiveInt


def load_config(path: str | Path = DEFAULT_CONFIG) -> ServerConfig:
    with Path(path).open(encoding="utf-8") as file:
        return ServerConfig.model_validate(yaml.safe_load(file))
