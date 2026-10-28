"""Weather and shadow corruptions in linear light."""
from __future__ import annotations
import cv2
import numpy as np
from crowdcc.base import Context, LevelParams
from crowdcc.crowd_spacing import crowd_spacing_px
from crowdcc.image_ops import (
    assert_shape_preserved,
    clip01,
    count_for_frame,
    depth_quintile_masks,
    directional_gradient_energy,
    gaussian_blur,
    linear_to_srgb,
    luminance,
    mean_luminance_ratio,
    srgb_to_linear,
    to_float01,
)
from crowdcc.registry import register

# Shared particle settings
MIN_PARTICLE_LENGTH_PX = 1.0
PARTICLE_SOFTEN_PX = 0.8
HEAD_RADIUS_FROM_SPACING = 0.5
HEAD_RADIUS_MIN_PX, HEAD_RADIUS_MAX_PX = 2.0, 24.0
HEAD_TOUCHED_FRACTION = 0.05
HEAD_OCCLUDED_FRACTION = 0.30

# rain_streaks
VEIL_LUMINANCE = 0.70
RAIN_VISIBLE_ADD = 0.04

# snow_particles
NEAR_FLAKE_STREAK_RATIO = 1.15
SIZE_LOGNORMAL_SIGMA = 0.40
SIZE_MULTIPLIER_MIN = 0.35
SIZE_MULTIPLIER_MAX = 2.5
VEIL_CONTRAST_GAIN = 1.5
VEIL_CONTRAST_LIFT = 0.5
FLAKE_LUMINANCE = 0.92
SNOW_VISIBLE_ALPHA = 0.50

# cast_shadow_illumination
SHADOW_WAVE_COMPONENTS = 3
SHADOW_WAVE_MIN_PERIODS, SHADOW_WAVE_MAX_PERIODS = 0.7, 2.5
SHADOW_COLOR_SHIFT = np.array([0.94, 1.0, 1.10], dtype=np.float32)


def draw_particles(
    shape_hw: tuple[int, int],
    count: int,
    length_px: np.ndarray,
    angle_deg: np.ndarray,
    thickness_px: np.ndarray,
    intensity: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw motion-streaked particles into a float32 layer."""
    height, width = shape_hw
    layer = np.zeros((height, width), dtype=np.float32)
    starts_x = rng.random(count, dtype=np.float32) * width
    starts_y = rng.random(count, dtype=np.float32) * height
    radians = np.deg2rad(angle_deg)
    # 0° is vertical; positive angles tilt right.
    steps_x = np.sin(radians) * length_px
    steps_y = np.cos(radians) * length_px
    drawn_width = np.maximum(1, np.round(thickness_px).astype(int))
    energy = intensity * (thickness_px / drawn_width)
    for i in range(count):
        start = (int(starts_x[i]), int(starts_y[i]))
        end = (int(starts_x[i] + steps_x[i]), int(starts_y[i] + steps_y[i]))
        cv2.line(layer, start, end, float(energy[i]),
                 thickness=int(drawn_width[i]), lineType=cv2.LINE_AA)
    return gaussian_blur(layer, PARTICLE_SOFTEN_PX)


def _particle_count(shape_hw: tuple[int, int], density: float) -> int:
    """Return aspect-scaled particle count."""
    return count_for_frame(shape_hw, density)


def _apply_veil(linear: np.ndarray, veil: float) -> np.ndarray:
    """Blend toward constant veil luminance."""
    return linear * (1.0 - veil) + veil * VEIL_LUMINANCE


def _apply_contrast_veil(linear: np.ndarray, veil: float) -> np.ndarray:
    """Apply ImageNet-C snow contrast veil in linear light."""
    if veil <= 0.0:
        return linear
    grey = luminance(linear)[..., None]
    washed = np.maximum(linear, grey * VEIL_CONTRAST_GAIN + VEIL_CONTRAST_LIFT)
    return linear * (1.0 - veil) + veil * washed


def head_visibility(
    particle_mask: np.ndarray, points: np.ndarray | None
) -> dict[str, float]:
    """Measure head coverage within crowd-scaled windows; NaN if no points."""
    if points is None or len(points) == 0:
        return {"heads_touched_fraction": float("nan"),
                "heads_occluded_fraction": float("nan"),
                "head_window_px": float("nan")}
    changed = particle_mask.astype(np.float32)
    spacing = crowd_spacing_px(points)
    radius = float(np.clip(HEAD_RADIUS_FROM_SPACING * spacing,
                           HEAD_RADIUS_MIN_PX, HEAD_RADIUS_MAX_PX)) \
        if np.isfinite(spacing) else HEAD_RADIUS_MIN_PX
    side = int(2 * round(radius) + 1)
    covered = cv2.boxFilter(changed, -1, (side, side), normalize=True)
    height, width = changed.shape
    cols = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
    per_head = covered[rows, cols]
    return {
        "heads_touched_fraction": float((per_head > HEAD_TOUCHED_FRACTION).mean()),
        "heads_occluded_fraction": float((per_head > HEAD_OCCLUDED_FRACTION).mean()),
        "head_window_px": float(side),
    }


def _oriented_anisotropy(delta: np.ndarray, angle_deg: float) -> float:
    """Return cross/along gradient energy ratio."""
    along, across = directional_gradient_energy(delta, angle_deg)
    return float(across / max(along, 1e-9))


def _verify_rain_streaks(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    assert_shape_preserved(src, dst, "rain_streaks")
    delta = (dst.astype(np.float32) - src.astype(np.float32)).mean(axis=2)
    angle = p.angle_deg
    return {
        "mean_luminance_ratio": mean_luminance_ratio(src, dst),
        "streak_anisotropy": _oriented_anisotropy(delta, angle),
    }


@register(
    name="rain_streaks",
    family="environment",
    gt_impact="preserving",
    needs=["gt_points"],
    levels={
        # Calibrated on JHU-Crowd++ training rain images.
        "light": LevelParams(density=300.0, length_frac=0.010, width_frac=0.0013,
                             angle_deg=14.0, angle_jitter_deg=5.0, opacity=0.25,
                             veil=0.030),
        "moderate": LevelParams(density=600.0, length_frac=0.015, width_frac=0.0013,
                                angle_deg=14.0, angle_jitter_deg=6.0, opacity=0.30,
                                veil=0.060),
        "high": LevelParams(density=900.0, length_frac=0.020, width_frac=0.0013,
                            angle_deg=14.0, angle_jitter_deg=7.0, opacity=0.35,
                            veil=0.090),
    },
    verify=_verify_rain_streaks,
)
def rain_streaks(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Add rain streaks and scattering veil; return image and metrics."""
    height, width = img.shape[:2]
    short_side = min(height, width)
    count = _particle_count((height, width), p.density)
    rng = ctx.rng_image
    lengths = np.maximum(p.length_frac * short_side * (0.6 + 0.8 * rng.random(count)),
                         MIN_PARTICLE_LENGTH_PX)
    angles = p.angle_deg + rng.normal(0.0, p.angle_jitter_deg, size=count)
    widths = np.full(count, p.width_frac * short_side, dtype=np.float32)
    intensity = 0.5 + 0.5 * rng.random(count)
    streaks = draw_particles((height, width), count, lengths, angles, widths,
                             intensity, rng)
    added = streaks * p.opacity
    linear = _apply_veil(srgb_to_linear(to_float01(img)), p.veil)
    out = linear_to_srgb(clip01(linear + added[..., None]))
    visible = added > RAIN_VISIBLE_ADD
    measurements = {"n_streaks": float(count),
                    "streak_length_px": float(np.median(lengths)),
                    "streak_pixel_fraction": float(visible.mean())}
    measurements.update(head_visibility(visible, ctx.gt_points))
    return out, measurements


def _verify_snow_particles(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    assert_shape_preserved(src, dst, "snow_particles")
    return {
        "mean_luminance_ratio": mean_luminance_ratio(src, dst),
        "white_pixel_gain": float((dst.min(axis=2) > 200).mean()
                                  - (src.min(axis=2) > 200).mean()),
    }


def _size_multipliers(count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate clipped, mean-1 lognormal flake sizes."""
    mu = -0.5 * SIZE_LOGNORMAL_SIGMA ** 2          # Preserve mean-1 lognormal scaling.
    return np.clip(rng.lognormal(mu, SIZE_LOGNORMAL_SIGMA, size=count),
                   SIZE_MULTIPLIER_MIN, SIZE_MULTIPLIER_MAX)


def _flake_layer(
    shape_hw: tuple[int, int], p: LevelParams, density: float, size_frac: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Render sharp far flakes and blurred near flakes."""
    height, width = shape_hw
    short_side = min(height, width)
    total = _particle_count(shape_hw, density)
    n_near = int(round(total * p.near_fraction))
    n_far = total - n_near
    def population(count: int, size_frac: float, blur_px: float,
                   streak_ratio: float) -> np.ndarray:
        if count <= 0:
            return np.zeros(shape_hw, dtype=np.float32)
        sizes = size_frac * short_side * _size_multipliers(count, rng)
        lengths = np.maximum(sizes * streak_ratio * (0.5 + rng.random(count)),
                             MIN_PARTICLE_LENGTH_PX)
        angles = p.angle_deg + rng.normal(0.0, p.angle_jitter_deg, size=count)
        layer = draw_particles(shape_hw, count, lengths, angles, sizes,
                               np.ones(count, dtype=np.float32), rng)
        return gaussian_blur(layer, blur_px)
    # Draw far flakes first for reproducible RNG order.
    far = population(n_far, size_frac, 0.0, p.streak_ratio)
    near = population(n_near, size_frac * p.near_size_ratio,
                      p.near_blur_frac * short_side, NEAR_FLAKE_STREAK_RATIO)
    return clip01(far + near), total


@register(
    name="snow_particles",
    family="environment",
    gt_impact="preserving",
    needs=["gt_points"],
    levels={
        # Calibrated on JHU-Crowd++ training snow images.
        "light": LevelParams(density=600.0, size_frac=0.00175, streak_ratio=1.5,
                             angle_deg=10.0, angle_jitter_deg=14.0,
                             near_fraction=0.03, near_size_ratio=3.5,
                             near_blur_frac=0.004, veil=0.08),
        "moderate": LevelParams(density=800.0, size_frac=0.0022, streak_ratio=1.6,
                                angle_deg=10.0, angle_jitter_deg=15.0,
                                near_fraction=0.045, near_size_ratio=4.0,
                                near_blur_frac=0.007, veil=0.13),
        "high": LevelParams(density=950.0, size_frac=0.0026, streak_ratio=1.7,
                            angle_deg=10.0, angle_jitter_deg=16.0,
                            near_fraction=0.055, near_size_ratio=4.5,
                            near_blur_frac=0.010, veil=0.17),
    },
    verify=_verify_snow_particles,
)
def snow_particles(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Composite snowflakes over a contrast veil."""
    density = p.density
    alpha, count = _flake_layer(img.shape[:2], p, density,
                                float(p.size_frac),
                                ctx.rng_image)
    alpha = np.maximum(alpha, np.rot90(alpha, k=2))
    linear = _apply_contrast_veil(srgb_to_linear(to_float01(img)),
                                  float(p.veil))
    alpha3 = alpha[..., None]
    out = linear_to_srgb(clip01(linear * (1.0 - alpha3) + alpha3 * FLAKE_LUMINANCE))
    visible = alpha > SNOW_VISIBLE_ALPHA
    measurements = {"n_flakes": float(count),
                    "flake_pixel_fraction": float(visible.mean())}
    measurements.update(head_visibility(visible, ctx.gt_points))
    return out, measurements


def scene_shadow_azimuth(linear: np.ndarray, rng: np.random.Generator) -> float:
    """Estimate shadow azimuth from luminance gradient with 12-degree jitter."""
    small = cv2.resize(luminance(linear), (32, 32), interpolation=cv2.INTER_AREA)
    grad_x = float(np.mean(np.gradient(small, axis=1)))
    grad_y = float(np.mean(np.gradient(small, axis=0)))
    azimuth = np.rad2deg(np.arctan2(-grad_y, -grad_x))
    return float(azimuth + rng.normal(0.0, 12.0))


def _shadow_alpha(
    shape_hw: tuple[int, int], p: LevelParams, azimuth_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a wavy shadow mask with smoothstep edges."""
    height, width = shape_hw
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radians = np.deg2rad(azimuth_deg)
    distance = xs * np.cos(radians) + ys * np.sin(radians)
    tangent = -xs * np.sin(radians) + ys * np.cos(radians)
    for _ in range(SHADOW_WAVE_COMPONENTS):
        periods = rng.uniform(SHADOW_WAVE_MIN_PERIODS, SHADOW_WAVE_MAX_PERIODS)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        distance = distance + (p.boundary_wave_amp / SHADOW_WAVE_COMPONENTS) * np.sin(
            periods * np.pi * tangent + phase
        )
    offset = float(np.quantile(distance, p.coverage))
    penumbra = max(p.penumbra_frac, 1e-4)
    edge = np.clip(0.5 - (distance - offset) / (2.0 * penumbra), 0.0, 1.0)
    return (edge * edge * (3.0 - 2.0 * edge)).astype(np.float32)


def _verify_cast_shadow_illumination(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    assert_shape_preserved(src, dst, "cast_shadow_illumination")
    src_lum = luminance(srgb_to_linear(to_float01(src)))
    dst_lum = luminance(srgb_to_linear(to_float01(dst)))
    ratio_map = dst_lum / np.maximum(src_lum, 1e-6)
    darkest = np.quantile(ratio_map, 0.10)
    brightest = np.quantile(ratio_map, 0.90)
    return {
        "mean_luminance_ratio": mean_luminance_ratio(src, dst),
        "umbra_transmission": float(darkest),
        "sunlit_transmission": float(brightest),
        "shadow_contrast": float(brightest / max(darkest, 1e-6)),
    }


@register(
    name="cast_shadow_illumination",
    family="environment",
    gt_impact="preserving",
    needs=[],
    levels={
        "light": LevelParams(coverage=0.30, umbra_transmission=0.62,
                             penumbra_frac=0.10, boundary_wave_amp=0.10),
        "moderate": LevelParams(coverage=0.45, umbra_transmission=0.38,
                                penumbra_frac=0.07, boundary_wave_amp=0.13),
        "high": LevelParams(coverage=0.60, umbra_transmission=0.20,
                            penumbra_frac=0.05, boundary_wave_amp=0.16),
    },
    verify=_verify_cast_shadow_illumination,
)
def cast_shadow_illumination(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply shadow attenuation and color shift."""
    linear = srgb_to_linear(to_float01(img))
    azimuth = scene_shadow_azimuth(linear, ctx.rng_image)
    alpha = _shadow_alpha(img.shape[:2], p, azimuth, ctx.rng_image)
    transmission = p.umbra_transmission
    gain = 1.0 - alpha * (1.0 - transmission)
    tint = 1.0 - alpha[..., None] * (1.0 - SHADOW_COLOR_SHIFT)
    out = linear_to_srgb(clip01(linear * gain[..., None] * tint))
    return out, {
        "shadow_azimuth_deg": float(azimuth),
        "shadowed_area_fraction": float((alpha > 0.5).mean()),
    }


# depth_aware_haze
HAZE_AIRLIGHT_TINT = np.array([0.97, 0.99, 1.04], dtype=np.float32)
HAZE_DEPTH_SMOOTH_FRAC = 0.004


def _smoothed_depth(depth: np.ndarray, frac: float) -> np.ndarray:
    short_side = float(min(depth.shape[:2]))
    sigma = max(frac * short_side, 0.0)
    if sigma <= 0:
        return depth
    return np.clip(gaussian_blur(depth, sigma), 0.0, 1.0)


def _verify_depth_aware_haze(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify greater extinction at far depths."""
    assert_shape_preserved(src, dst, "depth_aware_haze")
    if ctx.depth is None:
        raise AssertionError("depth_aware_haze verified without a depth map")
    beta = float(p.beta)
    depth = _smoothed_depth(ctx.depth, HAZE_DEPTH_SMOOTH_FRAC)
    extinction = 1.0 - np.exp(-beta * depth)
    near_mask, far_mask = depth_quintile_masks(depth)
    near_ext = float(extinction[near_mask].mean())
    far_ext = float(extinction[far_mask].mean())
    if not far_ext > near_ext:
        raise AssertionError(
            f"depth_aware_haze: far-quintile extinction {far_ext:.4f} "
            f"not above near-quintile {near_ext:.4f}"
        )
    src_lin = srgb_to_linear(to_float01(src))
    dst_lin = srgb_to_linear(to_float01(dst))
    change = np.abs(luminance(dst_lin) - luminance(src_lin))
    near_change = float(change[near_mask].mean())
    far_change = float(change[far_mask].mean())
    return {
        "extinction_near_quintile": near_ext,
        "extinction_far_quintile": far_ext,
        "extinction_far_over_near": far_ext / max(near_ext, 1e-6),
        "transmission_min": float(np.exp(-beta * float(depth.max()))),
        "transmission_mean": float(np.exp(-beta * depth).mean()),
        "pixel_change_near_quintile": near_change,
        "pixel_change_far_quintile": far_change,
        "pixel_change_far_over_near": far_change / max(near_change, 1e-6),
    }


@register(
    name="depth_aware_haze",
    family="environment",
    gt_impact="preserving",
    needs=["depth"],
    levels={
        "light": LevelParams(beta=0.18, airlight=0.80),
        "moderate": LevelParams(beta=0.32, airlight=0.85),
        "high": LevelParams(beta=0.55, airlight=0.90),
    },
    verify=_verify_depth_aware_haze,
)
def depth_aware_haze(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply Koschmieder haze: I = J*t + A*(1-t), t = exp(-beta*depth)."""
    if ctx.depth is None:
        raise RuntimeError(
            "depth_aware_haze requires a depth map. "
            "Generate it with crowdcc depth."
        )
    beta = float(p.beta)
    airlight_level = float(p.airlight)
    depth = _smoothed_depth(ctx.depth, HAZE_DEPTH_SMOOTH_FRAC)
    linear = srgb_to_linear(to_float01(img))
    if depth.shape != linear.shape[:2]:
        raise RuntimeError(
            f"depth map {depth.shape} does not match image {linear.shape[:2]}"
        )
    transmission = np.exp(-beta * depth).astype(np.float32)[..., None]
    airlight = np.clip(airlight_level * HAZE_AIRLIGHT_TINT, 0.0, 1.0)
    hazed = linear * transmission + airlight[None, None, :] * (1.0 - transmission)
    near_mask, far_mask = depth_quintile_masks(depth)
    extinction = 1.0 - transmission[..., 0]
    return linear_to_srgb(clip01(hazed)), {
        "beta": beta,
        "airlight": airlight_level,
        "transmission_mean": float(transmission.mean()),
        "transmission_far_quintile": float(transmission[..., 0][far_mask].mean()),
        "transmission_near_quintile": float(transmission[..., 0][near_mask].mean()),
        "extinction_far_over_near": float(
            extinction[far_mask].mean() / max(extinction[near_mask].mean(), 1e-6)
        ),
    }
