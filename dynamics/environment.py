from dataclasses import dataclass, field
from pathlib import Path

from .force_models import ForceModelConfig
from .frames import FrameConfig


@dataclass(slots=True)
class EnvironmentConfig:
    """Настройка окружения для backend распространения."""

    central_body: str = "Earth"
    frame: FrameConfig = field(default_factory=FrameConfig)
    force_models: ForceModelConfig = field(default_factory=ForceModelConfig)
    space_weather_file: str | Path | None = None
    gravity_coefficients_file: str | Path | None = None
    force_scale_factors: dict[str, float] = field(default_factory=dict)
