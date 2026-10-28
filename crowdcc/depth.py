"""Load cached depth maps for haze and defocus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEPTH_DIR_NAME = "_depth"
DEPTH_DTYPE = np.float16
DEPTH_MANIFEST_NAME = "manifest.json"

RANGE_TOLERANCE = 1e-3


def depth_dir(out_root: Path | str, dataset: str, split: str) -> Path:
    """Return the depth cache directory for a dataset split."""
    return Path(out_root) / dataset / split / DEPTH_DIR_NAME


def load_depth_dir(
    directory: Path | str,
    image_id: str,
    expect_hw: tuple[int, int] | None = None,
    *,
    what: str | None = None,
    rebuild_hint: str = "",
) -> np.ndarray:
    """Load a cached depth map (0=near, 1=far), validating shape and range."""
    path = Path(directory) / f"{image_id}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"no depth map for {what or image_id} at {path}. "
            f"Run: crowdcc depth {rebuild_hint}"
        )
    depth = np.load(path).astype(np.float32)

    if depth.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D map, got shape {depth.shape}")
    if expect_hw is not None and depth.shape != tuple(expect_hw):
        raise ValueError(
            f"{path}: depth {depth.shape} does not match image {tuple(expect_hw)}"
        )
    if not np.all(np.isfinite(depth)):
        raise ValueError(f"{path}: depth contains NaN/Inf")
    low, high = float(depth.min()), float(depth.max())
    if low < -RANGE_TOLERANCE or high > 1.0 + RANGE_TOLERANCE:
        raise ValueError(
            f"{path}: depth range [{low:.4f}, {high:.4f}] is outside [0, 1]; the "
            f"cache was written with a different normalization"
        )
    return np.clip(depth, 0.0, 1.0)


def load_depth_manifest_dir(directory: Path | str) -> dict[str, Any]:
    """Load the depth cache manifest."""
    path = Path(directory) / DEPTH_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no depth manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
