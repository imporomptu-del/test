"""Background subtraction and noise-normalized residual products."""

from .model import (
    BackgroundConfig,
    NoiseConfig,
    PreprocessingError,
    RobustPreprocessor,
)
from .types import ResidualFrame

__all__ = [
    "BackgroundConfig",
    "NoiseConfig",
    "PreprocessingError",
    "ResidualFrame",
    "RobustPreprocessor",
]
