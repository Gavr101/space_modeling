from dataclasses import dataclass, field
from pathlib import Path

from .force_models import ForceModelConfig
from .frames import FrameConfig


@dataclass(slots=True)
class EnvironmentConfig:
    """Environment setup for propagation backend."""

    central_body: str = "Earth"
    frame: FrameConfig = field(default_factory=FrameConfig)
    force_models: ForceModelConfig = field(default_factory=ForceModelConfig)
    space_weather_file: str | Path | None = None
    gravity_coefficients_file: str | Path | None = None
