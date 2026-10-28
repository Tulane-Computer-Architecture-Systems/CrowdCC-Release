"""CrowdCC crowd-counting corruption benchmark."""

from crowdcc.base import (
    LEVELS,
    Context,
    CorruptionSpec,
    GTTransform,
    LevelParams,
)
from crowdcc.registry import all_specs, get, register, resolve

__version__ = "1.0.0"

__all__ = [
    "LEVELS",
    "Context",
    "CorruptionSpec",
    "GTTransform",
    "LevelParams",
    "__version__",
    "all_specs",
    "get",
    "register",
    "resolve",
]
