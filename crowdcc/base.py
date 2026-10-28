"""Core data types for CrowdCC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import numpy as np

LEVELS: tuple[str, ...] = ("light", "moderate", "high")

GTImpact = Literal["preserving", "altering"]

KNOWN_NEEDS: frozenset[str] = frozenset({"gt_points", "depth", "ffmpeg"})


class LevelParams:
    """Schema-free severity parameters with attribute access."""

    def __init__(self, **params: Any) -> None:
        self._params: dict[str, Any] = dict(params)

    def __getattr__(self, name: str) -> Any:
        # Avoid recursion during unpickling.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._params[name]
        except KeyError:
            raise AttributeError(
                f"level parameter {name!r} not defined; available: "
                f"{sorted(self._params)}"
            ) from None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable parameters."""
        return {k: _jsonable(v) for k, v in self._params.items()}

    def get(self, name: str, default: Any = None) -> Any:
        return self._params.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LevelParams) and self._params == other._params

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"LevelParams({inner})"


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    return v


@dataclass(frozen=True)
class Context:
    """Per-image inputs with shared sensor and per-image RNGs."""

    image_id: str
    dataset: str
    level: str
    rng_sensor: np.random.Generator
    rng_image: np.random.Generator
    gt_points: np.ndarray | None = None       # (N, 2) float (x, y)
    depth: np.ndarray | None = None           # (H, W) float32 in [0, 1]; 0 = near, 1 = far
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GTTransform:
    """Warp or mask GT points, with optional occlusion removal."""

    kind: Literal["homography", "dropped_mask", "occluded_mask"]
    output_hw: tuple[int, int]
    homography: np.ndarray | None = None       # (3, 3) float64
    dropped_mask: np.ndarray | None = None     # (N,) bool, True = removed
    occluded_mask: np.ndarray | None = None    # (N,) bool, True = occluded
    drop_occluded: bool = False

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return in-frame points and the original-point keep mask."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        moved = points
        if self.kind == "homography":
            if self.homography is None:
                raise ValueError("kind='homography' requires a homography matrix")
            moved = apply_homography(self.homography, points)

        kept = np.ones(len(points), dtype=bool)
        if self.dropped_mask is not None:
            kept &= ~np.asarray(self.dropped_mask, dtype=bool)
        if self.occluded_mask is not None and self.drop_occluded:
            kept &= ~np.asarray(self.occluded_mask, dtype=bool)

        height, width = self.output_hw
        in_frame = (
            (moved[:, 0] >= 0) & (moved[:, 0] < width)
            & (moved[:, 1] >= 0) & (moved[:, 1] < height)
        )
        kept &= in_frame
        return moved[kept], kept


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)   # Avoid division by zero at the horizon.
    return projected[:, :2] / w


@dataclass(frozen=True)
class CorruptionSpec:
    """Registered corruption function and metadata."""

    name: str
    family: str
    gt_impact: GTImpact
    needs: tuple[str, ...]
    levels: Mapping[str, LevelParams]
    fn: Callable[..., Any]
    verify: Callable[..., dict[str, float]] | None = None
    # Verification returns metrics or raises AssertionError.

    def level_names(self) -> tuple[str, ...]:
        """Return levels in severity order."""
        return tuple(lv for lv in LEVELS if lv in self.levels)

    def params_for(self, level: str) -> LevelParams:
        try:
            return self.levels[level]
        except KeyError:
            raise KeyError(
                f"corruption {self.name!r} has no level {level!r}; "
                f"defined: {list(self.levels)}"
            ) from None
