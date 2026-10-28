"""Foreground occlusion, FOV loss, and perspective shifts."""

from __future__ import annotations

import cv2
import numpy as np

from crowdcc.base import Context, GTTransform, LevelParams, apply_homography
from crowdcc.image_ops import (
    assert_shape_preserved,
    clip01,
    gaussian_blur,
    linear_to_srgb,
    luminance,
    srgb_to_linear,
    texture_energy,
    to_float01,
)
from crowdcc.registry import register
from crowdcc.seeds import image_seed
from crowdcc.scene_ops import (
    OCCLUSION_ALPHA,
    _fill_polygons,
    person_silhouette,
    points_inside_mask,
)

# structured_foreground_occlusion
OCCLUSION_LAYOUT_KEY = "structured_foreground_occlusion/shared-layout"

OCCLUDER_EDGE_BLUR_FRAC = 0.0025

OCCLUDER_MAX_SHAPES = 40

OCCLUDER_CORE_ALPHA = 0.95

OCCLUDER_CLEAR_ALPHA = 0.02

OCCLUDER_CORE_ERODE_BLURS = 1.0

OCCLUDER_MAX_TEXTURE_RETENTION = 0.70

OCCLUDER_MIN_BACKGROUND_RETENTION = 0.95

# viewpoint_perspective_shift
WARP_FILL = 0.0


def _eroded(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Erode a boolean mask by a pixel radius."""
    kernel = np.ones((2 * int(pixels) + 1, 2 * int(pixels) + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _draw_pole(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a full-height leaning pole."""
    height, width = mask.shape
    x = float(rng.uniform(0.05, 0.95)) * width
    # Preserve an opaque core after blur and erosion.
    thickness = float(rng.uniform(0.014, 0.030)) * min(height, width)
    lean = float(rng.normal(0.0, 0.02)) * width
    cv2.fillPoly(mask, [np.round(np.array(
        [[x - thickness / 2, 0], [x + thickness / 2, 0],
         [x + lean + thickness / 2, height], [x + lean - thickness / 2, height]]
    )).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)


def _draw_railing(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a railing in the lower frame."""
    height, width = mask.shape
    top = float(rng.uniform(0.55, 0.80)) * height
    bar_h = float(rng.uniform(0.016, 0.030)) * height
    for row in (top, top + float(rng.uniform(0.10, 0.18)) * height):
        cv2.rectangle(mask, (0, int(row)), (width, int(row + bar_h)), 1.0, -1)
    post_w = max(int(0.014 * width), 1)
    spacing = max(int(float(rng.uniform(0.10, 0.20)) * width), post_w * 3)
    for x in range(int(rng.uniform(0, spacing)), width, spacing):
        cv2.rectangle(mask, (x, int(top)), (x + post_w, height), 1.0, -1)


def _draw_banner(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a skewed banner."""
    height, width = mask.shape
    w = float(rng.uniform(0.18, 0.40)) * width
    h = float(rng.uniform(0.10, 0.25)) * height
    x = float(rng.uniform(0.0, 1.0)) * (width - w)
    y = float(rng.uniform(0.0, 0.6)) * (height - h)
    skew = float(rng.normal(0.0, 0.06)) * h
    cv2.fillPoly(mask, [np.round(np.array(
        [[x, y], [x + w, y - skew], [x + w, y + h - skew], [x, y + h]]
    )).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)


def _draw_umbrella(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Draw an umbrella canopy and handle."""
    height, width = mask.shape
    radius = float(rng.uniform(0.10, 0.22)) * min(height, width)
    cx = float(rng.uniform(0.1, 0.9)) * width
    cy = float(rng.uniform(0.25, 0.85)) * height
    cv2.ellipse(mask, (int(cx), int(cy)), (int(radius), int(radius * 0.62)),
                0, 180, 360, 1.0, -1)
    handle_w = max(int(0.012 * min(height, width)), 1)
    cv2.rectangle(mask, (int(cx - handle_w / 2), int(cy)),
                  (int(cx + handle_w / 2), int(cy + radius * 1.1)), 1.0, -1)


def _draw_spectator(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a bottom-anchored spectator silhouette."""
    height, width = mask.shape
    person_h = float(rng.uniform(0.45, 0.95)) * height
    x = float(rng.uniform(0.05, 0.95)) * width
    _fill_polygons(mask, person_silhouette(person_h, rng), (x, float(height)))


_OCCLUDER_KINDS = {
    "pole": _draw_pole,
    "railing": _draw_railing,
    "banner": _draw_banner,
    "umbrella": _draw_umbrella,
    "spectator": _draw_spectator,
}


def occluder_mask(
    shape_hw: tuple[int, int], p: LevelParams, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate occluders up to the coverage or shape limit."""
    height, width = shape_hw
    mask = np.zeros(shape_hw, dtype=np.float32)
    kinds = [k for k in p.occluder_mix if k in _OCCLUDER_KINDS]
    if not kinds:
        raise ValueError(f"no known occluder kinds in {p.occluder_mix!r}")

    target = float(p.coverage)
    drawn: list[str] = []
    for _ in range(OCCLUDER_MAX_SHAPES):
        if float(mask.mean()) >= target:
            break
        kind = kinds[int(rng.integers(len(kinds)))]
        _OCCLUDER_KINDS[kind](mask, rng)
        drawn.append(kind)
    return np.clip(mask, 0.0, 1.0), {
        "n_occluders": float(len(drawn)),
        "occluder_coverage": float((mask > OCCLUSION_ALPHA).mean()),
        "occluder_coverage_target": target,
    }


def _shared_occluder_rng(ctx: Context) -> np.random.Generator:
    """Seed a shared occluder layout by image and level."""
    return np.random.default_rng(
        image_seed(ctx.image_id, OCCLUSION_LAYOUT_KEY, ctx.level)
    )


def _render_occluders(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Render occluders and return image, alpha, and metrics."""
    linear = srgb_to_linear(to_float01(img))
    mask, record = occluder_mask(img.shape[:2], p, _shared_occluder_rng(ctx))
    edge = OCCLUDER_EDGE_BLUR_FRAC * float(min(img.shape[:2]))
    alpha = clip01(gaussian_blur(mask, edge))

    darkness = float(p.occluder_darkness)
    radiance = darkness * float(np.median(luminance(linear)))
    occluded = linear * (1.0 - alpha[..., None]) + radiance * alpha[..., None]
    record["occluder_radiance"] = radiance
    record["occluder_coverage"] = float((alpha > OCCLUSION_ALPHA).mean())
    record["occluder_core_fraction"] = float((alpha > OCCLUDER_CORE_ALPHA).mean())
    return linear_to_srgb(clip01(occluded)), alpha, record


def _verify_occlusion(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify texture loss under occluders and background retention."""
    assert_shape_preserved(src, dst, "structured_foreground_occlusion")
    _, alpha, record = _render_occluders(src, p, ctx)
    solid = alpha > OCCLUSION_ALPHA
    edge_px = OCCLUDER_EDGE_BLUR_FRAC * float(min(src.shape[:2]))
    erode_px = max(int(round(OCCLUDER_CORE_ERODE_BLURS * edge_px)), 1)
    core = _eroded(alpha > OCCLUDER_CORE_ALPHA, erode_px)
    if not solid.any():
        raise AssertionError("structured_foreground_occlusion: no occluders placed")
    if not core.any():
        raise AssertionError(
            "structured_foreground_occlusion: no opaque core after blur and erosion"
        )

    # Check texture away from alpha edges.
    clear = _eroded(alpha < OCCLUDER_CLEAR_ALPHA, erode_px)
    inside_texture = texture_energy(dst, core) / max(texture_energy(src, core), 1e-9)
    outside_texture = (texture_energy(dst, clear) / max(texture_energy(src, clear), 1e-9)
                       if clear.any() else float("nan"))
    if not inside_texture < OCCLUDER_MAX_TEXTURE_RETENTION:
        raise AssertionError(
            f"structured_foreground_occlusion: core texture retention "
            f"{inside_texture:.4f} (max {OCCLUDER_MAX_TEXTURE_RETENTION})"
        )
    if clear.any() and not outside_texture > OCCLUDER_MIN_BACKGROUND_RETENTION:
        raise AssertionError(
            f"structured_foreground_occlusion: background texture retention "
            f"{outside_texture:.4f} (min {OCCLUDER_MIN_BACKGROUND_RETENTION})"
        )

    src_lum = luminance(srgb_to_linear(to_float01(src)))
    dst_lum = luminance(srgb_to_linear(to_float01(dst)))
    occluded = points_inside_mask(ctx.gt_points, alpha)
    n_points = 0 if ctx.gt_points is None else len(ctx.gt_points)
    return {
        **record,
        "texture_retention_under_occluders": inside_texture,
        "texture_retention_outside": outside_texture,
        "luminance_retention_under_occluders": float(
            dst_lum[core].mean() / max(src_lum[core].mean(), 1e-9)),
        "luminance_retention_outside": float(
            dst_lum[clear].mean() / max(src_lum[clear].mean(), 1e-9)) if clear.any()
        else float("nan"),
        "n_heads": float(n_points),
        "n_heads_occluded": float(int(occluded.sum())),
        "heads_occluded_fraction": float(occluded.mean()) if n_points else 0.0,
    }


_OCCLUSION_LEVELS = {
    "light": LevelParams(coverage=0.06, occluder_darkness=0.18,
                         occluder_mix=("pole", "railing", "banner")),
    "moderate": LevelParams(coverage=0.15, occluder_darkness=0.16,
                            occluder_mix=("pole", "railing", "banner", "umbrella")),
    "high": LevelParams(coverage=0.28, occluder_darkness=0.14,
                        occluder_mix=("pole", "railing", "banner", "umbrella",
                                      "spectator")),
}


@register(
    name="structured_foreground_occlusion",
    family="visibility_loss",
    gt_impact="altering",
    needs=["gt_points"],
    levels=_OCCLUSION_LEVELS,
    verify=_verify_occlusion,
)
def structured_foreground_occlusion(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Add foreground occluders and mask covered GT heads."""
    out, alpha, _ = _render_occluders(img, p, ctx)
    return out, GTTransform(
        kind="occluded_mask", output_hw=img.shape[:2],
        occluded_mask=points_inside_mask(ctx.gt_points, alpha),
        drop_occluded=True,
    )


def keystone_homography(
    shape_hw: tuple[int, int], top_compression: float
) -> np.ndarray:
    """Build a top-compressing, bottom-expanding homography."""
    height, width = shape_hw
    inset = 0.5 * float(top_compression) * width
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                      dtype=np.float32)
    target = np.array([[inset, 0], [width - inset, 0],
                       [width + inset, height], [-inset, height]], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target).astype(np.float64)


def assert_homography_roundtrip(
    matrix: np.ndarray, points: np.ndarray | None, tolerance: float = 1e-4
) -> float:
    """Check forward/inverse warp error against tolerance."""
    if points is None or len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    warped = apply_homography(matrix, pts)
    back = apply_homography(np.linalg.inv(matrix), warped)
    error = float(np.abs(back - pts).max())
    if not error <= tolerance:
        raise AssertionError(
            f"homography round-trip error {error:.3e} px exceeds {tolerance:.0e}"
        )
    return error


def _verify_viewpoint_perspective_shift(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify the keystone warp and vacated image region."""
    assert_shape_preserved(src, dst, "viewpoint_perspective_shift")
    compression = float(p.top_compression)
    matrix = keystone_homography(src.shape[:2], compression)
    error = assert_homography_roundtrip(matrix, ctx.gt_points)

    filled = (to_float01(dst).max(axis=2) <= 1e-6)
    height = dst.shape[0]
    top_share = float(filled[: height // 2].sum()) / max(float(filled.sum()), 1.0)
    if compression > 0 and not filled.any():
        raise AssertionError("viewpoint_perspective_shift: no vacated region")
    return {
        "top_compression": compression,
        "homography_roundtrip_max_px": error,
        "vacated_fraction": float(filled.mean()),
        "vacated_share_in_top_half": top_share,
    }


@register(
    name="viewpoint_perspective_shift",
    family="visibility_loss",
    gt_impact="altering",
    needs=["gt_points"],
    levels={
        "light": LevelParams(top_compression=0.15),
        "moderate": LevelParams(top_compression=0.30),
        "high": LevelParams(top_compression=0.45),
    },
    verify=_verify_viewpoint_perspective_shift,
)
def viewpoint_perspective_shift(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Apply a keystone warp to the image and GT points."""
    height, width = img.shape[:2]
    compression = float(p.top_compression)
    matrix = keystone_homography((height, width), compression)
    assert_homography_roundtrip(matrix, ctx.gt_points)

    warped = cv2.warpPerspective(
        to_float01(img), matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(WARP_FILL,) * 3,
    )
    return clip01(warped), GTTransform(
        kind="homography", output_hw=(height, width), homography=matrix,
    )


# fov_loss_camera_reaim
_FOV_DIRECTIONS = {
    "left": (1, 0, 0, 0), "right": (0, 1, 0, 0),
    "top": (0, 0, 1, 0), "bottom": (0, 0, 0, 1),
    "corner": (1, 0, 1, 0),
}


def fov_window(
    shape_hw: tuple[int, int], crop_fraction: float, direction: str
) -> tuple[int, int, int, int]:
    """Return half-open bounds of the retained image window."""
    height, width = shape_hw
    try:
        left, right, top, bottom = _FOV_DIRECTIONS[direction]
    except KeyError:
        raise KeyError(
            f"unknown fov direction {direction!r}; known: {sorted(_FOV_DIRECTIONS)}"
        ) from None
    share = crop_fraction / 2.0 if (left + right + top + bottom) > 1 else crop_fraction
    return (int(round(left * share * width)), int(round(top * share * height)),
            width - int(round(right * share * width)),
            height - int(round(bottom * share * height)))


def _verify_fov_loss(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify blanked edges and unchanged retained pixels."""
    assert_shape_preserved(src, dst, "fov_loss_camera_reaim")
    fraction = float(p.crop_fraction)
    direction = str(p.direction)
    x0, y0, x1, y1 = fov_window(src.shape[:2], fraction, direction)

    retained = np.zeros(src.shape[:2], dtype=bool)
    retained[y0:y1, x0:x1] = True
    lost = ~retained
    if not lost.any():
        raise AssertionError(
            f"fov_loss_camera_reaim: no field of view lost at crop_fraction={fraction}"
        )
    if not np.array_equal(src[retained], dst[retained]):
        raise AssertionError("fov_loss_camera_reaim: retained region modified")
    lost_luminance = float(to_float01(dst)[lost].max()) if lost.any() else 0.0
    if lost_luminance > 1e-6:
        raise AssertionError(
            f"fov_loss_camera_reaim: lost region not blank (max {lost_luminance:.4f})"
        )

    dropped = _fov_dropped_mask(ctx.gt_points, (x0, y0, x1, y1))
    n_points = 0 if ctx.gt_points is None else len(ctx.gt_points)
    return {
        "crop_fraction": fraction,
        "fov_lost_area_fraction": float(lost.mean()),
        "output_padded": 1.0,
        "n_heads": float(n_points),
        "n_heads_lost": float(int(dropped.sum())),
        "heads_lost_fraction": float(dropped.mean()) if n_points else 0.0,
    }


def _fov_dropped_mask(
    points: np.ndarray | None, window: tuple[int, int, int, int]
) -> np.ndarray:
    """Mark GT points outside the retained window."""
    if points is None or len(points) == 0:
        return np.zeros(0, dtype=bool)
    x0, y0, x1, y1 = window
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inside = ((pts[:, 0] >= x0) & (pts[:, 0] < x1)
              & (pts[:, 1] >= y0) & (pts[:, 1] < y1))
    return ~inside


@register(
    name="fov_loss_camera_reaim",
    family="visibility_loss",
    gt_impact="altering",
    needs=["gt_points"],
    levels={
        "light": LevelParams(crop_fraction=0.15, direction="right"),
        "moderate": LevelParams(crop_fraction=0.30, direction="corner"),
        "high": LevelParams(crop_fraction=0.45, direction="corner"),
    },
    verify=_verify_fov_loss,
)
def fov_loss_camera_reaim(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Blank frame edges and drop out-of-window GT points."""
    fraction = float(p.crop_fraction)
    direction = str(p.direction)
    x0, y0, x1, y1 = fov_window(img.shape[:2], fraction, direction)

    out = np.zeros(img.shape, dtype=np.float32)
    out[y0:y1, x0:x1] = to_float01(img)[y0:y1, x0:x1]
    return out, GTTransform(
        kind="dropped_mask", output_hw=img.shape[:2],
        dropped_mask=_fov_dropped_mask(ctx.gt_points, (x0, y0, x1, y1)),
    )
