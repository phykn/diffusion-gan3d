"""LR generation and separate SR refinement; the web backend uses LR.

LR reads the run's train.yaml plus config/gen.yaml generation defaults.
SR reads settings embedded in its weight export. Training resume uses the
training checkpoint's settings.
"""

from src.anchor import PlaneAnchor
from src.predict.inference import LowResolutionAPI
from src.predict.sr.extension import extend_hr
from src.predict.sr.inference import SuperResolutionAPI

__all__ = [
    "LowResolutionAPI",
    "PlaneAnchor",
    "SuperResolutionAPI",
    "extend_hr",
]
