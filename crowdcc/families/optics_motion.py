"""Blur, lens contamination, and flare corruptions."""

from __future__ import annotations

import cv2
import numpy as np

from crowdcc.base import Context, GTTransform, LevelParams
from crowdcc.crowd_spacing import resolve_spacing_px
from crowdcc.image_ops import (
    assert_shape_preserved,
    clip01,
    count_for_frame,
    depth_quintile_masks,
    directional_gradient_energy,
    gaussian_blur,
    layered_blur_composite,
    linear_to_srgb,
    luminance,
    points_inside_mask,
    srgb_to_linear,
    texture_energy,
    to_float01,
)
from crowdcc.registry import register

# far_field_defocus
LAYER_FEATHER_FRAC = 0.003

MAX_SIGMA_SHORT_SIDE_FRAC = 0.05


def layer_sigmas(
    n_layers: int, focus_depth: float, max_sigma_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return depth-bin centres and blur sigmas based on distance from focus."""
    edges = np.linspace(0.0, 1.0, int(n_layers) + 1, dtype=np.float32)
    centres = 0.5 * (edges[:-1] + edges[1:])
    reach = max(abs(1.0 - focus_depth), abs(focus_depth), 1e-6)
    sigmas = max_sigma_px * np.abs(centres - focus_depth) / reach
    return centres, sigmas.astype(np.float32)


def _resolve_max_sigma(p: LevelParams, ctx: Context, shape_hw: tuple[int, int]) -> tuple[float, float]:
    """Return blur sigma limit and crowd spacing in pixels."""
    spacing = resolve_spacing_px(ctx.gt_points, shape_hw)
    frac = float(p.max_blur_spacing_frac)
    ceiling = MAX_SIGMA_SHORT_SIDE_FRAC * float(min(shape_hw))
    return min(frac * spacing, ceiling), spacing


def _verify_far_field_defocus(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify greater far-field blur; report pixel change without asserting."""
    assert_shape_preserved(src, dst, "far_field_defocus")
    if ctx.depth is None:
        raise AssertionError("far_field_defocus verified without a depth map")

    depth = ctx.depth
    max_sigma, _ = _resolve_max_sigma(p, ctx, src.shape[:2])
    focus = float(p.focus_depth)
    n_layers = int(p.layers)
    _, sigmas = layer_sigmas(n_layers, focus, max_sigma)
    feather = LAYER_FEATHER_FRAC * float(min(src.shape[:2]))
    _, sigma_map = layered_blur_composite(
        srgb_to_linear(to_float01(src)), depth, sigmas, feather
    )

    near_mask, far_mask = depth_quintile_masks(depth)
    near_sigma = float(sigma_map[near_mask].mean())
    far_sigma = float(sigma_map[far_mask].mean())
    if not far_sigma > near_sigma:
        raise AssertionError(
            f"far_field_defocus: far-quintile sigma {far_sigma:.3f} px "
            f"not above near-quintile {near_sigma:.3f} px"
        )

    src_f = to_float01(src)
    dst_f = to_float01(dst)
    change = np.abs(dst_f - src_f).mean(axis=2)
    return {
        "blur_sigma_near_quintile_px": near_sigma,
        "blur_sigma_far_quintile_px": far_sigma,
        "blur_sigma_far_over_near": far_sigma / max(near_sigma, 1e-6),
        "blur_sigma_max_px": float(sigmas.max()),
        "pixel_change_near_quintile": float(change[near_mask].mean()),
        "pixel_change_far_quintile": float(change[far_mask].mean()),
    }


@register(
    name="far_field_defocus",
    family="optics_motion",
    gt_impact="preserving",
    needs=["depth", "gt_points"],
    levels={
        "light": LevelParams(
            max_blur_spacing_frac=0.05, focus_depth=0.08, layers=2),
        "moderate": LevelParams(
            max_blur_spacing_frac=0.10, focus_depth=0.08, layers=3),
        "high": LevelParams(
            max_blur_spacing_frac=0.15, focus_depth=0.08, layers=4),
    },
    verify=_verify_far_field_defocus,
)
def far_field_defocus(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply depth-binned blur with feathered masks in linear light."""
    if ctx.depth is None:
        raise RuntimeError(
            "far_field_defocus requires a depth map. "
            "Generate it with `crowdcc depth`."
        )

    linear = srgb_to_linear(to_float01(img))
    depth = ctx.depth
    if depth.shape != linear.shape[:2]:
        raise RuntimeError(
            f"depth map {depth.shape} does not match image {linear.shape[:2]}"
        )

    max_sigma, spacing = _resolve_max_sigma(p, ctx, img.shape[:2])
    focus = float(p.focus_depth)
    n_layers = int(p.layers)
    _, sigmas = layer_sigmas(n_layers, focus, max_sigma)
    feather = LAYER_FEATHER_FRAC * float(min(img.shape[:2]))

    out, sigma_map = layered_blur_composite(linear, depth, sigmas, feather)
    near_mask, far_mask = depth_quintile_masks(depth)
    return linear_to_srgb(clip01(out)), {
        "crowd_spacing_px": float(spacing),
        "max_blur_sigma_px": float(max_sigma),
        "focus_depth": focus,
        "layers": float(n_layers),
        "blur_sigma_near_quintile_px": float(sigma_map[near_mask].mean()),
        "blur_sigma_far_quintile_px": float(sigma_map[far_mask].mean()),
        "blur_sigma_mean_px": float(sigma_map.mean()),
    }


# camera_motion_blur
KERNEL_SAMPLES_PER_PX = 8

MAX_MOTION_LENGTH_SHORT_SIDE_FRAC = 0.12

MOTION_MIN_ANISOTROPY_MARGIN = 0.15

# lens_contamination
LOCAL_BACKGROUND_MIN_RETENTION = 0.95
LOCAL_FOREGROUND_MAX_RETENTION = 0.97

MASK_INSIDE_ALPHA = 0.60

MIN_DROP_RADIUS_PX = 3.0
MAX_DROP_RADIUS_FRAC = 0.20

DROP_RIM_START = 0.82
DROP_RIM_DARKEN = 0.55
DROP_SPECULAR_OFFSET = 0.38
DROP_SPECULAR_RADIUS = 0.13
DROP_SPECULAR_GAIN = 0.28
DROP_SAG_RANGE = (1.0, 1.35)

CONTAM_TOUCH_THRESHOLD = 2.0 / 255.0

CONTAM_MAX_TOUCHED_FRACTION = 0.50

CONTAM_GT_ALTERING_OPACITY = 0.8

CONTAM_MIN_SOURCE_TEXTURE = 0.0164

# lens_flare_veiling
FLARE_SOURCE_X_RANGE = (-0.25, 1.25)
FLARE_SOURCE_Y_RANGE = (-0.25, 0.45)

FLARE_TINT = np.array([1.00, 0.92, 0.80], dtype=np.float32)

GHOST_AXIS_STEP = 0.45
GHOST_TINTS = np.array([[0.55, 0.75, 1.00],
                        [1.00, 0.80, 0.55],
                        [0.70, 1.00, 0.75]], dtype=np.float32)

FLARE_SEARCH_GRID = 64


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Return a normalized motion kernel; 0° down, 90° right."""
    length = max(float(length_px), 1.0)
    radius = int(np.ceil(0.5 * length))
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float64)

    n_samples = int(np.ceil(length * KERNEL_SAMPLES_PER_PX)) + 1
    offsets = np.linspace(-0.5 * length, 0.5 * length, n_samples)
    radians = np.deg2rad(angle_deg)
    xs = radius + offsets * np.sin(radians)
    ys = radius + offsets * np.cos(radians)

    left = np.floor(xs).astype(int)
    top = np.floor(ys).astype(int)
    frac_x, frac_y = xs - left, ys - top
    for step_y, weight_y in ((0, 1.0 - frac_y), (1, frac_y)):
        for step_x, weight_x in ((0, 1.0 - frac_x), (1, frac_x)):
            np.add.at(kernel,
                      (np.clip(top + step_y, 0, size - 1),
                       np.clip(left + step_x, 0, size - 1)),
                      weight_y * weight_x)
    return (kernel / kernel.sum()).astype(np.float32)


def apply_motion_blur(linear: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
    """Apply motion blur in linear light."""
    kernel = motion_kernel(length_px, angle_deg)
    blurred = cv2.filter2D(np.asarray(linear, dtype=np.float32), -1, kernel,
                           borderType=cv2.BORDER_REFLECT101)
    return blurred.astype(np.float32)


def _resolve_motion(
    p: LevelParams, ctx: Context, shape_hw: tuple[int, int]
) -> tuple[float, float, float]:
    """Resolve motion length, clipped angle jitter, and crowd spacing."""
    spacing = resolve_spacing_px(ctx.gt_points, shape_hw)
    frac = float(p.length_spacing_frac)
    ceiling = MAX_MOTION_LENGTH_SHORT_SIDE_FRAC * float(min(shape_hw))
    length = min(frac * spacing, ceiling)

    mean_angle = float(p.angle_deg)
    jitter = float(p.angle_jitter_deg)
    offset = float(np.clip(ctx.rng_image.normal(0.0, jitter),
                           -p.angle_clip_sigmas * jitter, p.angle_clip_sigmas * jitter))
    return length, mean_angle + offset, spacing


def _verify_camera_motion_blur(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify directional gradient loss using normalized retention."""
    assert_shape_preserved(src, dst, "camera_motion_blur")
    angle = float(p.angle_deg)
    src_along, src_across = directional_gradient_energy(to_float01(src).mean(axis=2), angle)
    dst_along, dst_across = directional_gradient_energy(to_float01(dst).mean(axis=2), angle)
    along_retention = dst_along / max(src_along, 1e-9)
    across_retention = dst_across / max(src_across, 1e-9)
    margin = (across_retention - along_retention) / max(1.0 - along_retention, 1e-9)

    if not margin > MOTION_MIN_ANISOTROPY_MARGIN:
        raise AssertionError(
            f"camera_motion_blur: anisotropy margin {margin:.3f} below "
            f"{MOTION_MIN_ANISOTROPY_MARGIN} (gradient retention {along_retention:.4f} "
            f"along, {across_retention:.4f} across)"
        )
    return {
        "gradient_retention_along_axis": along_retention,
        "gradient_retention_across_axis": across_retention,
        "motion_anisotropy_margin": margin,
        "motion_anisotropy_ratio": across_retention / max(along_retention, 1e-9),
    }


@register(
    name="camera_motion_blur",
    family="optics_motion",
    gt_impact="preserving",
    needs=["gt_points"],
    levels={
        "light": LevelParams(length_spacing_frac=0.05, angle_deg=90.0,
                             angle_jitter_deg=8.0, angle_clip_sigmas=2.5),
        "moderate": LevelParams(length_spacing_frac=0.15, angle_deg=90.0,
                                angle_jitter_deg=8.0, angle_clip_sigmas=2.5),
        "high": LevelParams(length_spacing_frac=0.25, angle_deg=90.0,
                            angle_jitter_deg=8.0, angle_clip_sigmas=2.5),
    },
    verify=_verify_camera_motion_blur,
)
def camera_motion_blur(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply crowd-scaled motion blur in linear light."""
    length, angle, spacing = _resolve_motion(p, ctx, img.shape[:2])
    linear = srgb_to_linear(to_float01(img))
    blurred = apply_motion_blur(linear, length, angle)

    return linear_to_srgb(clip01(blurred)), {
        "crowd_spacing_px": float(spacing),
        "motion_length_px": float(length),
        "motion_angle_deg": float(angle),
        "motion_length_over_spacing": float(length / max(spacing, 1e-6)),
    }


def _droplet_geometry(
    shape_hw: tuple[int, int], p: LevelParams, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate droplet centres and lognormal radii."""
    height, width = shape_hw
    short_side = float(min(height, width))
    count = count_for_frame(shape_hw, float(p.density))
    centres = np.column_stack([rng.random(count) * width, rng.random(count) * height])
    scale = float(p.radius_frac) * short_side
    radii = scale * np.exp(rng.normal(0.0, float(p.radius_lognormal_sigma), size=count))
    return centres, np.clip(radii, MIN_DROP_RADIUS_PX, MAX_DROP_RADIUS_FRAC * short_side)


def _render_droplet(
    canvas: np.ndarray, source: np.ndarray, centre: np.ndarray, radius: float,
    p: LevelParams, alpha_total: np.ndarray, opacity: float = 1.0,
    occluding: np.ndarray | None = None,
) -> None:
    """Composite a refracted droplet and update coverage masks."""
    height, width = canvas.shape[:2]
# Deterministic sag; no RNG draws.
    lo, hi = DROP_SAG_RANGE
    sag = lo + (hi - lo) * float(np.modf(centre[0] * 0.6180339887 + centre[1] * 0.3141592653)[0])
    ry = radius * sag
    x0, y0 = max(int(centre[0] - radius) - 1, 0), max(int(centre[1] - ry) - 1, 0)
    x1, y1 = min(int(centre[0] + radius) + 2, width), min(int(centre[1] + ry) + 2, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = (xx - centre[0]) / radius, (yy - centre[1]) / ry
    normalized = np.clip(np.sqrt(dx * dx + dy * dy), 0.0, 1.0)
    gain = 1.0 - float(p.refraction_strength) * (1.0 - normalized ** 2)

    map_x = (centre[0] + (xx - centre[0]) * gain - x0).astype(np.float32)
    map_y = (centre[1] + (yy - centre[1]) * gain - y0).astype(np.float32)
    patch = np.ascontiguousarray(source[y0:y1, x0:x1])
    refracted = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
    refracted = gaussian_blur(refracted, float(p.blur_radius_frac) * radius)
    refracted = refracted + float(p.brighten)

    rim = np.clip((normalized - DROP_RIM_START) / max(1.0 - DROP_RIM_START, 1e-4), 0.0, 1.0)
    refracted = refracted * (1.0 - DROP_RIM_DARKEN * (rim ** 2))[..., None]

    sx = centre[0] - DROP_SPECULAR_OFFSET * radius
    sy = centre[1] - DROP_SPECULAR_OFFSET * ry
    srad = max(DROP_SPECULAR_RADIUS * radius, 0.8)
    spec = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * srad * srad)))
    refracted = refracted + DROP_SPECULAR_GAIN * spec[..., None] * (normalized < 1.0)[..., None]

    edge = float(p.edge_feather_frac)
    ramp = np.clip((1.0 - normalized) / max(edge, 1e-4), 0.0, 1.0)
    alpha = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)

    composite = alpha * float(opacity)
    canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1.0 - composite[..., None])
                            + refracted * composite[..., None])
    np.maximum(alpha_total[y0:y1, x0:x1], alpha, out=alpha_total[y0:y1, x0:x1])
    if occluding is not None and opacity > CONTAM_GT_ALTERING_OPACITY:
        np.maximum(occluding[y0:y1, x0:x1], alpha, out=occluding[y0:y1, x0:x1])


def _verify_lens_contamination(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify droplet texture loss and background retention."""
    assert_shape_preserved(src, dst, "lens_contamination")
    change = np.abs(to_float01(dst) - to_float01(src)).max(axis=2)
    touched = change > CONTAM_TOUCH_THRESHOLD
    touched_fraction = float(touched.mean())

    if not touched_fraction < CONTAM_MAX_TOUCHED_FRACTION:
        raise AssertionError(
            f"lens_contamination: {touched_fraction:.1%} of pixels altered "
            f"(max {CONTAM_MAX_TOUCHED_FRACTION:.0%})"
        )
    if not touched.any():
        raise AssertionError("lens_contamination: no droplets placed")

    source_texture = texture_energy(src, touched)
    inside_retention = texture_energy(dst, touched) / max(source_texture, 1e-9)
    clear = ~touched
    outside_retention = (texture_energy(dst, clear) / max(texture_energy(src, clear), 1e-9)
                         if clear.any() else float("nan"))

    if clear.any() and not outside_retention > LOCAL_BACKGROUND_MIN_RETENTION:
        raise AssertionError(
            f"lens_contamination: background texture retention "
            f"{outside_retention:.4f} (min {LOCAL_BACKGROUND_MIN_RETENTION})"
        )
    applicable = source_texture >= CONTAM_MIN_SOURCE_TEXTURE
    if applicable and not inside_retention < LOCAL_FOREGROUND_MAX_RETENTION:
        raise AssertionError(
            f"lens_contamination: texture retention under droplets "
            f"{inside_retention:.4f} (max {LOCAL_FOREGROUND_MAX_RETENTION})"
        )
    return {
        "texture_retention_under_drops": inside_retention,
        "texture_retention_on_clear_glass": outside_retention,
        "droplet_touched_fraction": touched_fraction,
        "source_texture_under_drops": source_texture,
        "destruction_check_applicable": float(applicable),
    }


@register(
    name="lens_contamination",
    family="optics_motion",
    gt_impact="altering",
    needs=["gt_points"],
    levels={
        "light": LevelParams(opacity_range=(0.4, 0.95), density=150.0,
                             radius_frac=0.01, radius_lognormal_sigma=0.3,
                             refraction_strength=0.3, blur_radius_frac=0.15,
                             brighten=0.03, edge_feather_frac=0.20),
        "moderate": LevelParams(opacity_range=(0.4, 0.95), density=65.0,
                                radius_frac=0.02, radius_lognormal_sigma=0.4,
                                refraction_strength=0.4, blur_radius_frac=0.20,
                                brighten=0.05, edge_feather_frac=0.15),
        "high": LevelParams(opacity_range=(0.4, 0.95), density=38.0,
                            radius_frac=0.03, radius_lognormal_sigma=0.5,
                            refraction_strength=0.5, blur_radius_frac=0.25,
                            brighten=0.07, edge_feather_frac=0.10),
    },
    verify=_verify_lens_contamination,
)
def lens_contamination(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Render lens droplets and mark occluded GT points."""
    linear = srgb_to_linear(to_float01(img))
    centres, radii = _droplet_geometry(img.shape[:2], p, ctx.rng_image)

# Sample opacity after geometry to preserve RNG order.
    lo, hi = (float(v) for v in p.opacity_range)
    opacities = ctx.rng_image.uniform(lo, hi, size=len(radii))

    canvas = linear.copy()
    alpha_total = np.zeros(img.shape[:2], dtype=np.float32)
    occluding = np.zeros(img.shape[:2], dtype=np.float32)
    for centre, radius, opacity in zip(centres, radii, opacities):
        _render_droplet(canvas, linear, centre, radius, p, alpha_total,
                        float(opacity), occluding)

    return linear_to_srgb(clip01(canvas)), GTTransform(
        kind="occluded_mask", output_hw=img.shape[:2],
        occluded_mask=points_inside_mask(ctx.gt_points, occluding, MASK_INSIDE_ALPHA),
        drop_occluded=True,
    )


def flare_source_xy(
    shape_hw: tuple[int, int], rng: np.random.Generator
) -> tuple[float, float]:
    """Sample glare source coordinates, including off-frame positions."""
    height, width = shape_hw
    x = float(rng.uniform(*FLARE_SOURCE_X_RANGE)) * width
    y = float(rng.uniform(*FLARE_SOURCE_Y_RANGE)) * height
    return x, y


def _glare_field(
    shape_hw: tuple[int, int], source_xy: tuple[float, float], p: LevelParams
) -> np.ndarray:
    """Generate Gaussian glare and inverse-square veiling."""
    height, width = shape_hw
    short_side = float(min(height, width))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.sqrt((xx - source_xy[0]) ** 2 + (yy - source_xy[1]) ** 2) / short_side

    core = float(p.core_gain) * np.exp(-(distance / float(p.glow_radius_frac)) ** 2)
    veil = float(p.veil_strength) / (1.0 + (distance / float(p.veil_radius_frac)) ** 2)
    return (core + veil).astype(np.float32)


def _ghost_field(
    shape_hw: tuple[int, int], source_xy: tuple[float, float], p: LevelParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render tinted aperture ghosts along the flare axis."""
    height, width = shape_hw
    short_side = float(min(height, width))
    ghosts = np.zeros((height, width, 3), dtype=np.float32)
    count = int(p.ghost_count)
    if count <= 0 or float(p.ghost_strength) <= 0:
        return ghosts

    centre = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    axis = centre - np.array(source_xy, dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for index in range(count):
        position = centre + axis * (GHOST_AXIS_STEP * (index + 1))
        radius = short_side * float(p.ghost_radius_frac) * (0.6 + 0.8 * float(rng.random()))
        distance = np.sqrt((xx - position[0]) ** 2 + (yy - position[1]) ** 2)
        disc = np.clip(1.0 - (distance / radius) ** 2, 0.0, 1.0)
        amplitude = float(p.ghost_strength) / (index + 1)
        ghosts += amplitude * disc[..., None] * GHOST_TINTS[index % len(GHOST_TINTS)]
    return ghosts


def _verify_lens_flare_veiling(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify positive, radially concentrated flare luminance."""
    assert_shape_preserved(src, dst, "lens_flare_veiling")
    src_lum = luminance(srgb_to_linear(to_float01(src)))
    dst_lum = luminance(srgb_to_linear(to_float01(dst)))
    added = dst_lum - src_lum
    if not float(added.mean()) > 0.0:
        raise AssertionError(
            f"lens_flare_veiling: mean added luminance {float(added.mean()):+.6f} "
            f"(expected > 0)"
        )

    coarse = cv2.resize(added, (FLARE_SEARCH_GRID, FLARE_SEARCH_GRID),
                        interpolation=cv2.INTER_AREA)
    row, col = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    height, width = added.shape
    peak = ((col + 0.5) / FLARE_SEARCH_GRID * width, (row + 0.5) / FLARE_SEARCH_GRID * height)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.sqrt((xx - peak[0]) ** 2 + (yy - peak[1]) ** 2)
    near = added[distance <= np.quantile(distance, 0.25)].mean()
    far = added[distance >= np.quantile(distance, 0.75)].mean()

    if not float(near) > float(far):
        raise AssertionError(
            f"lens_flare_veiling: added luminance not radially concentrated "
            f"(near {float(near):.6f}, far {float(far):.6f})"
        )
    return {
        "added_luminance_mean": float(added.mean()),
        "added_luminance_near_source": float(near),
        "added_luminance_far_from_source": float(far),
        "flare_radial_concentration": float(near / max(far, 1e-9)),
        "estimated_source_x": float(peak[0]),
        "estimated_source_y": float(peak[1]),
    }


@register(
    name="lens_flare_veiling",
    family="optics_motion",
    gt_impact="preserving",
    needs=[],
    levels={
        "light": LevelParams(core_gain=0.45, glow_radius_frac=0.12, veil_strength=0.070,
                             veil_radius_frac=0.55, ghost_count=2, ghost_strength=0.010,
                             ghost_radius_frac=0.07),
        "moderate": LevelParams(core_gain=0.70, glow_radius_frac=0.17, veil_strength=0.130,
                                veil_radius_frac=0.62, ghost_count=3, ghost_strength=0.018,
                                ghost_radius_frac=0.08),
        "high": LevelParams(core_gain=1.20, glow_radius_frac=0.24, veil_strength=0.240,
                            veil_radius_frac=0.72, ghost_count=3, ghost_strength=0.028,
                            ghost_radius_frac=0.09),
    },
    verify=_verify_lens_flare_veiling,
)
def lens_flare_veiling(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Add glare, veiling, and reflection ghosts in linear light."""
    source = flare_source_xy(img.shape[:2], ctx.rng_image)
    glare = _glare_field(img.shape[:2], source, p)
    ghosts = _ghost_field(img.shape[:2], source, p, ctx.rng_image)

    linear = srgb_to_linear(to_float01(img))
    flared = linear + glare[..., None] * FLARE_TINT[None, None, :] + ghosts

    return linear_to_srgb(clip01(flared)), {
        "source_x": float(source[0]),
        "source_y": float(source[1]),
        "source_in_frame": float(0 <= source[0] < img.shape[1]
                                 and 0 <= source[1] < img.shape[0]),
        "added_radiance_mean": float(glare.mean()),
        "added_radiance_max": float(glare.max()),
        "clipped_pixel_fraction": float((flared.max(axis=2) >= 1.0).mean()),
    }
