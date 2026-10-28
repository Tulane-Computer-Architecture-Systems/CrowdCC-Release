"""Sensor and exposure corruptions in linear light."""

from __future__ import annotations

import numpy as np

from crowdcc.base import Context, LevelParams
from crowdcc.image_ops import (
    assert_shape_preserved,
    clip01,
    gaussian_blur,
    linear_to_srgb,
    luminance,
    mean_luminance_ratio,
    normal_field,
    poisson_field,
    srgb_to_linear,
    to_float01,
    to_uint8,
)
from crowdcc.registry import register

# low_light
MIN_MEAN_LINEAR_LUMINANCE = 0.010

MAX_EFFECTIVE_EXPOSURE = 0.75


# highlight_clipping_bloom
BLOOM_SIGMA_MAX_PX = 48.0

# fixed_pattern_sensor_noise
SENSOR_MAX_DIM = 12288

STUCK_POOL_SIZE = 200_000

HOT_PIXEL_SHARE = 0.5

# vignetting_lens_shading
MIN_TRANSMISSION = 0.05

# gaussian_noise, low_light
NOISE_BINS = 8
NOISE_BIN_LO, NOISE_BIN_HI = 1, NOISE_BINS - 2

NOISE_SAMPLE_CAP = 1_000_000


def _noise_std_bright_over_dark(src: np.ndarray, dst: np.ndarray) -> float:
    """Report bright/dark residual-noise ratio without assertions."""
    src_lin = srgb_to_linear(to_float01(src))
    dst_lin = srgb_to_linear(to_float01(dst))
    residual = (dst_lin - src_lin).reshape(-1)
    signal = src_lin.reshape(-1)

    stride = max(1, residual.size // NOISE_SAMPLE_CAP)
    residual = residual[::stride]
    signal = signal[::stride]

    edges = np.quantile(signal, np.linspace(0.0, 1.0, NOISE_BINS + 1))
    bin_index = np.clip(np.searchsorted(edges, signal, side="right") - 1, 0, NOISE_BINS - 1)
    stds = np.array([
        float(residual[bin_index == b].std()) if np.any(bin_index == b) else np.nan
        for b in (NOISE_BIN_LO, NOISE_BIN_HI)
    ])
    if not np.all(np.isfinite(stds)) or stds[0] < 1e-9:
        return float("nan")
    return float(stds[1] / stds[0])


def _verify_gaussian_noise(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    assert_shape_preserved(src, dst, "gaussian_noise")
    src_lin = srgb_to_linear(to_float01(src))
    dst_lin = srgb_to_linear(to_float01(dst))
    return {
        "residual_std_linear": float((dst_lin - src_lin).std()),
        "noise_std_bright_over_dark": _noise_std_bright_over_dark(src, dst),
    }


@register(
    name="gaussian_noise",
    family="sensor_exposure",
    gt_impact="preserving",
    needs=[],
    levels={
        "light": LevelParams(sigma=0.010),
        "moderate": LevelParams(sigma=0.025),
        "high": LevelParams(sigma=0.050),
    },
    verify=_verify_gaussian_noise,
)
def gaussian_noise(img: np.ndarray, p: LevelParams, ctx: Context) -> np.ndarray:
    """Add Gaussian read noise in linear light."""
    sigma = p.sigma
    linear = srgb_to_linear(to_float01(img))
    read_noise = normal_field(ctx.rng_image, linear.shape, sigma)
    return linear_to_srgb(clip01(linear + read_noise))


def _scene_adaptive_exposure(
    linear: np.ndarray, exposure: float
) -> tuple[float, float]:
    """Apply the scene brightness floor, capped by MAX_EFFECTIVE_EXPOSURE."""
    mean_lum = float(luminance(linear).mean())
    floor_exposure = MIN_MEAN_LINEAR_LUMINANCE / max(mean_lum, 1e-6)
    effective = float(np.clip(max(exposure, floor_exposure),
                              0.0, MAX_EFFECTIVE_EXPOSURE))
    return effective, mean_lum


def _verify_low_light(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Verify output luminance does not exceed input."""
    assert_shape_preserved(src, dst, "low_light")
    ratio = mean_luminance_ratio(src, dst)
    if ratio > 1.0:
        raise AssertionError(f"low_light: luminance ratio {ratio:.4f} > 1")
    return {"mean_luminance_ratio": ratio}


@register(
    name="low_light",
    family="sensor_exposure",
    gt_impact="preserving",
    needs=[],
    levels={
        "light": LevelParams(exposure=0.60, photons=1200.0, read_sigma=0.0015),
        "moderate": LevelParams(exposure=0.40, photons=800.0, read_sigma=0.0030),
        "high": LevelParams(exposure=0.20, photons=400.0, read_sigma=0.0060),
    },
    verify=_verify_low_light,
)
def low_light(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Reduce exposure; add Poisson shot noise and Gaussian read noise."""
    requested = p.exposure
    linear = srgb_to_linear(to_float01(img))
    effective, mean_lum = _scene_adaptive_exposure(linear, requested)

    photons = float(p.photons)
    read_sigma = float(p.read_sigma)
    reduced = linear * effective
    shot = poisson_field(ctx.rng_image, reduced * photons) / photons
    noisy = shot + normal_field(ctx.rng_image, linear.shape, read_sigma)
    out = linear_to_srgb(clip01(noisy))

    return out, {
        "exposure_effective": effective,
        "exposure_requested": float(requested),
        "scene_mean_linear_luminance": mean_lum,
        "exposure_floored": float(effective > requested),
        "exposure_capped": float(effective >= MAX_EFFECTIVE_EXPOSURE),
    }


def _verify_highlight_clipping_bloom(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    assert_shape_preserved(src, dst, "highlight_clipping_bloom")
    src_clipped = float((src.max(axis=2) >= 255).mean())
    dst_clipped = float((dst.max(axis=2) >= 255).mean())
    return {
        "mean_luminance_ratio": mean_luminance_ratio(src, dst),
        "clipped_fraction": dst_clipped,
        "clipped_fraction_gain": dst_clipped - src_clipped,
    }


@register(
    name="highlight_clipping_bloom",
    family="sensor_exposure",
    gt_impact="preserving",
    needs=[],
    levels={
        # Severity: clipped_fraction_gain, not SSIM.
        "light": LevelParams(exposure=2.4, bloom_sigma_frac=0.010, bloom_gain=0.35),
        "moderate": LevelParams(exposure=3.2, bloom_sigma_frac=0.018, bloom_gain=0.55),
        "high": LevelParams(exposure=3.8, bloom_sigma_frac=0.026, bloom_gain=0.72),
    },
    verify=_verify_highlight_clipping_bloom,
)
def highlight_clipping_bloom(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Increase exposure and bloom excess highlights before clipping."""
    linear = srgb_to_linear(to_float01(img))
    exposed = linear * p.exposure
    excess = np.maximum(exposed - 1.0, 0.0)
    base = clip01(exposed)

    short_side = min(img.shape[0], img.shape[1])
    bloom_sigma = min(p.bloom_sigma_frac * short_side, BLOOM_SIGMA_MAX_PX)
    halation = gaussian_blur(excess, bloom_sigma) * p.bloom_gain
    out = linear_to_srgb(clip01(base + halation))

    return out, {
        "bloom_sigma_px": float(bloom_sigma),
        "excess_energy_fraction": float(excess.mean() / max(linear.mean(), 1e-9)),
    }


def fixed_pattern_offset(
    shape_hw: tuple[int, int], column_sigma: float, row_sigma: float,
    rng_sensor: np.random.Generator,
) -> np.ndarray:
    """Generate sensor-fixed row and column offsets."""
    height, width = shape_hw
    if max(height, width) > SENSOR_MAX_DIM:
        raise ValueError(
            f"image {height}x{width} exceeds the modelled sensor "
            f"({SENSOR_MAX_DIM} px); raise SENSOR_MAX_DIM and regenerate"
        )
    columns = normal_field(rng_sensor, (SENSOR_MAX_DIM,), column_sigma)
    rows = normal_field(rng_sensor, (SENSOR_MAX_DIM,), row_sigma)
    return (columns[None, :width] + rows[:height, None]).astype(np.float32)


def stuck_pixels(
    shape_hw: tuple[int, int], stuck_fraction: float, rng_sensor: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sensor-fixed hot/dead pixels shared by equal-sized frames."""
    height, width = shape_hw
    n_defects = int(round(stuck_fraction * height * width))
    if n_defects > STUCK_POOL_SIZE:
        raise ValueError(
            f"{n_defects} defects requested but the sensor pool holds "
            f"{STUCK_POOL_SIZE}; raise STUCK_POOL_SIZE and regenerate"
        )
    u = rng_sensor.random(STUCK_POOL_SIZE, dtype=np.float32)
    v = rng_sensor.random(STUCK_POOL_SIZE, dtype=np.float32)
    channel = rng_sensor.integers(0, 3, size=STUCK_POOL_SIZE)
    hot = rng_sensor.random(STUCK_POOL_SIZE, dtype=np.float32) < HOT_PIXEL_SHARE

    rows = np.minimum((v[:n_defects] * height).astype(np.int64), height - 1)
    cols = np.minimum((u[:n_defects] * width).astype(np.int64), width - 1)
    values = np.where(hot[:n_defects], 255, 0).astype(np.uint8)
    return rows, cols, channel[:n_defects].astype(np.int64), values


def _verify_fixed_pattern_sensor_noise(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Report residual sensor-noise metrics without pattern assertions."""
    assert_shape_preserved(src, dst, "fixed_pattern_sensor_noise")
    residual = dst.astype(np.float32) - src.astype(np.float32)
    column_means = residual.mean(axis=(0, 2))
    return {
        "column_pattern_std": float(column_means.std()),
        "residual_std": float(residual.std()),
        "extreme_pixel_fraction": float(((dst == 0) | (dst == 255)).mean()),
    }


@register(
    name="fixed_pattern_sensor_noise",
    family="sensor_exposure",
    gt_impact="preserving",
    needs=[],
    levels={
        "light": LevelParams(stuck_fraction=0.00005, column_sigma=0.0020,
                             row_sigma=0.0020, temporal_sigma=0.0020),
        "moderate": LevelParams(stuck_fraction=0.00020, column_sigma=0.0080,
                                row_sigma=0.0080, temporal_sigma=0.0080),
        "high": LevelParams(stuck_fraction=0.00030, column_sigma=0.0120,
                            row_sigma=0.0120, temporal_sigma=0.0120),
    },
    verify=_verify_fixed_pattern_sensor_noise,
)
def fixed_pattern_sensor_noise(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Add sensor-fixed defects and per-image read noise."""
    linear = srgb_to_linear(to_float01(img))
    offset = fixed_pattern_offset(
        img.shape[:2], p.column_sigma, p.row_sigma, ctx.rng_sensor
    )
    temporal = normal_field(ctx.rng_image, linear.shape, p.temporal_sigma)

    patterned = clip01(linear + offset[..., None] + temporal)
    out = to_uint8(linear_to_srgb(patterned))

    rows, cols, channels, values = stuck_pixels(
        img.shape[:2], p.stuck_fraction, ctx.rng_sensor
    )
    out[rows, cols, channels] = values
    return out, {"n_stuck_pixels": float(len(rows))}


def _normalized_radius_sq(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return squared radius normalized to 1 at image corners."""
    height, width = shape_hw
    half_h = max((height - 1) / 2.0, 1.0)
    half_w = max((width - 1) / 2.0, 1.0)
    y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / half_h
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / half_w
    return ((y[:, None] ** 2 + x[None, :] ** 2) / 2.0).astype(np.float32)


def _verify_vignetting_lens_shading(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    assert_shape_preserved(src, dst, "vignetting_lens_shading")
    radius_sq = _normalized_radius_sq(src.shape[:2])
    edge = radius_sq >= 0.5
    core = radius_sq <= 0.1
    src_lin = luminance(srgb_to_linear(to_float01(src)))
    dst_lin = luminance(srgb_to_linear(to_float01(dst)))
    before = float(src_lin[edge].mean()) / max(float(src_lin[core].mean()), 1e-9)
    after = float(dst_lin[edge].mean()) / max(float(dst_lin[core].mean()), 1e-9)
    corner = max(1.0 - p.k, MIN_TRANSMISSION)
    return {
        "edge_center_ratio": after,
        "edge_center_ratio_change": after / max(before, 1e-9),
        "corner_transmission": float(corner),
    }


@register(
    name="vignetting_lens_shading",
    family="sensor_exposure",
    gt_impact="preserving",
    needs=[],
    levels={
        # Exaggerated lens falloff.
        "light": LevelParams(k=1.25),
        "moderate": LevelParams(k=1.80),
        "high": LevelParams(k=2.70),
    },
    verify=_verify_vignetting_lens_shading,
)
def vignetting_lens_shading(img: np.ndarray, p: LevelParams, ctx: Context) -> np.ndarray:
    """Apply radial falloff max(1 - k*r², MIN_TRANSMISSION)."""
    k = p.k
    linear = srgb_to_linear(to_float01(img))
    shading = np.maximum(1.0 - k * _normalized_radius_sq(img.shape[:2]), MIN_TRANSMISSION)
    return linear_to_srgb(linear * shading[..., None])
