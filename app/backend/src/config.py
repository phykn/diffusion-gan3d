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
    max_tiles: PositiveInt


def load_server_config(path=DEFAULT_CONFIG):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ServerConfig.model_validate(values)
