"""Crowd-spacing estimates from GT points."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SPACING_K = 3

SPACING_CLIP_MIN_PX = 2.0
SPACING_CLIP_MAX_PX = 300.0

FALLBACK_SPACING_SHORT_SIDE_FRAC = 0.04
MIN_POINTS_FOR_SPACING = 3


def point_spacing_px(points: np.ndarray, k: int = SPACING_K) -> np.ndarray:
    """Return mean distance to up to k nearest neighbors per point."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.empty(0, dtype=np.float64)
    k_eff = min(k, len(pts) - 1)
    # First neighbor is the query point.
    distances, _ = cKDTree(pts).query(pts, k=k_eff + 1)
    return np.asarray(distances[:, 1:], dtype=np.float64).mean(axis=1)


def crowd_spacing_px(points: np.ndarray, k: int = SPACING_K) -> float:
    """Return clipped median crowd spacing; NaN for fewer than two points."""
    per_point = point_spacing_px(points, k=k)
    if per_point.size == 0:
        return float("nan")
    return float(np.clip(np.median(per_point), SPACING_CLIP_MIN_PX, SPACING_CLIP_MAX_PX))


def resolve_spacing_px(points: np.ndarray | None, shape_hw: tuple[int, int]) -> float:
    """Return spacing or short-side fallback for fewer than three heads."""
    fallback = FALLBACK_SPACING_SHORT_SIDE_FRAC * float(min(shape_hw))
    if points is None or len(points) < MIN_POINTS_FOR_SPACING:
        return fallback
    spacing = float(crowd_spacing_px(points))
    return spacing if np.isfinite(spacing) and spacing > 0 else fallback


def dataset_median_spacing_px(points_per_image: list[np.ndarray], k: int = SPACING_K) -> float:
    """Return median spacing across images for train-split calibration."""
    values = np.array([crowd_spacing_px(p, k=k) for p in points_per_image], dtype=np.float64)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("no image in the sample has a defined crowd spacing")
    return float(np.median(valid))


def head_diameter_px(box_wh: np.ndarray) -> np.ndarray:
    """Return sqrt(width * height) for JHU head boxes."""
    boxes = np.asarray(box_wh, dtype=np.float64).reshape(-1, 2)
    widths = np.maximum(boxes[:, 0], 0.0)
    heights = np.maximum(boxes[:, 1], 0.0)
    return np.sqrt(widths * heights)


SPACING_OVER_HEAD_DIAMETER = 1.75


def head_extent_px(
    points: np.ndarray, box_wh: np.ndarray | None = None, k: int = SPACING_K
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (head extents, spacing proxies, box fraction), per head."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    per_point = point_spacing_px(pts, k=k)
    # Keep one NaN proxy per head if spacing is undefined.
    proxy = (per_point / SPACING_OVER_HEAD_DIAMETER if per_point.size == len(pts)
             else np.full(len(pts), np.nan, dtype=np.float64))
    if len(pts) == 0 or box_wh is None:
        return proxy, proxy, 0.0

    boxes = np.asarray(box_wh, dtype=np.float64).reshape(-1, 2)
    if len(boxes) != len(pts):
        return proxy, proxy, 0.0
    measured = head_diameter_px(boxes)
    usable = np.isfinite(measured) & (measured > 0)
    extent = np.where(usable, measured, proxy)
    return extent, proxy, float(usable.mean())
