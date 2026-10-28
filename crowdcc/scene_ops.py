"""Geometry helpers for scene clutter and visibility loss."""

from __future__ import annotations

import cv2
import numpy as np

from crowdcc.image_ops import points_inside_mask as _points_inside_mask

OCCLUSION_ALPHA = 0.5


def points_inside_mask(points: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    """Flag GT points where the mask exceeds OCCLUSION_ALPHA."""
    return _points_inside_mask(points, mask, OCCLUSION_ALPHA)


def person_silhouette(
    height_px: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Build person-part polygons with feet at (0, 0) and y upward."""
    head_r = 0.11 * height_px
    shoulder = 0.30 * height_px * (0.9 + 0.2 * rng.random())
    hip = 0.20 * height_px * (0.9 + 0.2 * rng.random())
    torso_top, torso_bottom = 0.78 * height_px, 0.45 * height_px
    lean = float(rng.normal(0.0, 0.03)) * height_px

    head = np.array([[lean + head_r * np.cos(t), 0.89 * height_px + head_r * np.sin(t)]
                     for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    torso = np.array([[lean - shoulder / 2, torso_top], [lean + shoulder / 2, torso_top],
                      [hip / 2, torso_bottom], [-hip / 2, torso_bottom]])
    stride = (0.05 + 0.10 * rng.random()) * height_px
    leg_w = 0.07 * height_px
    legs = [np.array([[-hip / 2, torso_bottom], [-hip / 2 + leg_w, torso_bottom],
                      [-stride + leg_w, 0.0], [-stride, 0.0]]),
            np.array([[hip / 2 - leg_w, torso_bottom], [hip / 2, torso_bottom],
                      [stride, 0.0], [stride - leg_w, 0.0]])]
    swing = (0.04 + 0.10 * rng.random()) * height_px
    arm_w = 0.055 * height_px
    arms = [np.array([[lean - shoulder / 2, torso_top],
                      [lean - shoulder / 2 + arm_w, torso_top],
                      [lean - shoulder / 2 - swing + arm_w, torso_bottom - 0.05 * height_px],
                      [lean - shoulder / 2 - swing, torso_bottom - 0.05 * height_px]]),
            np.array([[lean + shoulder / 2 - arm_w, torso_top],
                      [lean + shoulder / 2, torso_top],
                      [lean + shoulder / 2 + swing, torso_bottom - 0.05 * height_px],
                      [lean + shoulder / 2 + swing - arm_w,
                       torso_bottom - 0.05 * height_px]])]
    return [head, torso, *legs, *arms]


def _fill_polygons(
    mask: np.ndarray, polygons: list[np.ndarray], origin_xy: tuple[float, float],
    scale: float = 1.0, flip_y: bool = True,
) -> None:
    """Rasterize local-frame polygons into the mask at origin_xy."""
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float64) * scale
        xs = pts[:, 0] + origin_xy[0]
        ys = origin_xy[1] - pts[:, 1] if flip_y else origin_xy[1] + pts[:, 1]
        cv2.fillPoly(mask, [np.round(np.column_stack([xs, ys])).astype(np.int32)], 1.0,
                     lineType=cv2.LINE_AA)
