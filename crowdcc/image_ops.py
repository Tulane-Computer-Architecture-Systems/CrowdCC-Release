"""Shared image operations for CrowdCC corruptions."""


from __future__ import annotations


from pathlib import Path


import cv2
import numpy as np


PNG_COMPRESSION_LEVEL = 6


_SRGB_LINEAR_CUTOFF = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_ALPHA = 0.055
_SRGB_GAMMA = 2.4
_LINEAR_CUTOFF = 0.0031308


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8 to float32 [0, 1], or cast float input."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.asarray(img, dtype=np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert [0, 1] floats to uint8 with clipping and half-up rounding."""
    scaled = np.asarray(img, dtype=np.float32) * 255.0
    return np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)


def clip01(img: np.ndarray) -> np.ndarray:
    """Clip values to [0, 1] as float32."""
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB [0, 1] to linear-light float32."""
    s = np.asarray(img, dtype=np.float32)
    low = s / _SRGB_LINEAR_SLOPE
    high = ((s + _SRGB_ALPHA) / (1.0 + _SRGB_ALPHA)) ** _SRGB_GAMMA
    return np.where(s <= _SRGB_LINEAR_CUTOFF, low, high).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear-light [0, 1] to sRGB float32."""
    lin = np.asarray(img, dtype=np.float32)
    low = lin * _SRGB_LINEAR_SLOPE
    high = (1.0 + _SRGB_ALPHA) * np.power(np.maximum(lin, 0.0), 1.0 / _SRGB_GAMMA) - _SRGB_ALPHA
    return np.where(lin <= _LINEAR_CUTOFF, low, high).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute Rec. 709 luminance from linear RGB."""
    values = np.asarray(rgb, dtype=np.float32)
    return (0.2126 * values[..., 0]
            + 0.7152 * values[..., 1]
            + 0.0722 * values[..., 2]).astype(np.float32)


def mean_luminance_ratio(src: np.ndarray, dst: np.ndarray) -> float:
    """Return the ratio of output to input mean linear luminance."""
    before = float(luminance(srgb_to_linear(to_float01(src))).mean())
    after = float(luminance(srgb_to_linear(to_float01(dst))).mean())
    return after / max(before, 1e-9)


def texture_energy(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute masked RMS grayscale gradient energy, or NaN if empty."""
    values = to_float01(np.asarray(img))
    grey = values.mean(axis=2) if values.ndim == 3 else values
    grad_y, grad_x = np.gradient(grey)
    energy = grad_x * grad_x + grad_y * grad_y
    selected = energy[mask] if mask is not None else energy
    if selected.size == 0:
        return float("nan")
    return float(np.sqrt(selected.mean()))


def assert_shape_preserved(src: np.ndarray, dst: np.ndarray, name: str) -> None:
    """Raise if a corruption changes the image shape."""
    if src.shape != dst.shape:
        raise AssertionError(f"{name} changed shape {src.shape} -> {dst.shape}")


DEPTH_QUANTILE_NEAR = 0.20
DEPTH_QUANTILE_FAR = 0.80


def depth_quintile_masks(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return near/far masks at the map's 20th/80th depth percentiles."""
    near_edge = float(np.quantile(depth, DEPTH_QUANTILE_NEAR))
    far_edge = float(np.quantile(depth, DEPTH_QUANTILE_FAR))
    return depth <= near_edge, depth >= far_edge


POISSON_BAND_ROWS = 256


def normal_field(
    rng: np.random.Generator, shape: tuple[int, ...], sigma: float
) -> np.ndarray:
    """Draw float32 Gaussian noise, or zeros for nonpositive sigma."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma)


def poisson_field(rng: np.random.Generator, expectation: np.ndarray) -> np.ndarray:
    """Draw float32 Poisson noise in row bands."""
    expectation = np.asarray(expectation, dtype=np.float32)
    out = np.empty(expectation.shape, dtype=np.float32)
    for start in range(0, expectation.shape[0], POISSON_BAND_ROWS):
        stop = min(start + POISSON_BAND_ROWS, expectation.shape[0])
        out[start:stop] = rng.poisson(expectation[start:stop]).astype(np.float32)
    return out


MIN_BLUR_SIGMA_PX = 0.3


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 3-sigma Gaussian blur, or cast to float32 for nonpositive sigma."""
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32)
    radius = int(np.ceil(3.0 * sigma))
    ksize = 2 * radius + 1
    blurred = cv2.GaussianBlur(
        np.asarray(img, dtype=np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    return blurred.astype(np.float32)


def count_for_frame(shape_hw: tuple[int, int], density: float) -> int:
    """Scale a square-frame element count by image aspect ratio."""
    height, width = shape_hw
    short_side, long_side = min(height, width), max(height, width)
    return int(round(density * long_side / short_side))


def directional_gradient_energy(
    grey: np.ndarray, angle_deg: float
) -> tuple[float, float]:
    """Return along/across RMS gradients; angles rotate from +y toward +x."""
    field = np.asarray(grey, dtype=np.float32)
    grad_x = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
    radians = np.deg2rad(angle_deg)
    along = np.sin(radians) * grad_x + np.cos(radians) * grad_y
    across = np.cos(radians) * grad_x - np.sin(radians) * grad_y
    return float(np.sqrt((along ** 2).mean())), float(np.sqrt((across ** 2).mean()))


def layered_blur_composite(
    image: np.ndarray,
    control: np.ndarray,
    sigmas: np.ndarray,
    feather_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite feathered blur layers and return image and sigma map."""
    n_bins = int(len(sigmas))
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    accum = np.zeros_like(image, dtype=np.float32)
    sigma_map = np.zeros(control.shape, dtype=np.float32)
    weight = np.zeros(control.shape, dtype=np.float32)

    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        # Include 1.0 in the final bin.
        mask = ((control >= lo) & (control < hi)) if k < n_bins - 1 else \
               ((control >= lo) & (control <= hi))
        mask = mask.astype(np.float32)
        if not mask.any():
            continue
        mask = gaussian_blur(mask, feather_px) if feather_px > 0 else mask
        sigma = float(sigmas[k])
        blurred = gaussian_blur(image, sigma) if sigma >= MIN_BLUR_SIGMA_PX else image
        accum += blurred * mask[..., None]
        sigma_map += sigma * mask
        weight += mask

    safe = np.maximum(weight, 1e-6)
    return accum / safe[..., None], sigma_map / safe


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Read uint8 RGB without EXIF rotation to preserve GT alignment."""
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png(img: np.ndarray) -> bytes:
    """Encode uint8 RGB as lossless PNG bytes."""
    if img.dtype != np.uint8:
        raise TypeError(f"PNG encode expects uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"PNG encode expects (H, W, 3) RGB, got shape {img.shape}")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION_LEVEL])
    if not ok:
        raise RuntimeError("cv2.imencode failed for PNG")
    return buf.tobytes()


def write_png(path: str | Path, img: np.ndarray) -> int:
    """Write RGB PNG and return the byte count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encode_png(img)
    path.write_bytes(payload)
    return len(payload)


def points_inside_mask(
    points: np.ndarray | None, mask: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Flag GT points whose mask value exceeds the threshold."""
    if points is None or len(points) == 0:
        return np.zeros(0, dtype=bool)
    height, width = mask.shape[:2]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cols = np.clip(np.round(pts[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(pts[:, 1]).astype(int), 0, height - 1)
    return mask[rows, cols] > threshold
