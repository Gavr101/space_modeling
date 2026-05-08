from dataclasses import dataclass, field

from .force_models import ForceModelConfig
from .frames import FrameConfig


@dataclass(slots=True)
class EnvironmentConfig:
    """Environment setup for propagation backend."""

    central_body: str = "Earth"
    frame: FrameConfig = field(default_factory=FrameConfig)
    force_models: ForceModelConfig = field(default_factory=ForceModelConfig)
