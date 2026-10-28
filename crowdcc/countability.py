"""Head-structure retention (HSR) metrics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .crowd_spacing import head_extent_px


HSR_PATCH_HEADS = 2.0

HSR_MIN_PATCH_PX = 9
HSR_MAX_PATCH_PX = 129

HSR_SIGMA_LO_HEADS = 1.0 / 8.0
HSR_SIGMA_HI_HEADS = 1.0 / 2.0
HSR_SIGMA_LO_MIN_PX = 0.6
HSR_SIGMA_HI_MIN_PX = 1.2

HSR_SCALE_BINS = 6


HSR_NOISE_FLOOR = 0.5

HSR_MIN_REF_DETECTABILITY = 1.0

HSR_TONE_BINS = 24
HSR_TONE_MIN_SAMPLES = 12

RETENTION_TAU = 0.5

_NOISE_KERNEL = np.array([[1.0, -2.0, 1.0],
                          [-2.0, 4.0, -2.0],
                          [1.0, -2.0, 1.0]], dtype=np.float32)
_NOISE_SCALE = float(np.sqrt(np.pi / 2.0) / 6.0)


def bandpass_noise_gain(sigma_lo: float, sigma_hi: float) -> float:
    """Return white-noise gain of a DoG band-pass filter."""
    a, b = float(sigma_lo) ** 2, float(sigma_hi) ** 2
    gain_sq = 1.0 / (4.0 * np.pi * a) + 1.0 / (4.0 * np.pi * b) - 1.0 / (np.pi * (a + b))
    return float(np.sqrt(max(gain_sq, 1e-12)))


@dataclass(frozen=True)
class HSRResult:
    """Per-head HSR scores and aggregate statistics."""

    hsr: np.ndarray                  # Scored heads; float64 in [0, 1]
    n_points: int
    tau: float = RETENTION_TAU

    @property
    def n_scored(self) -> int:
        return int(self.hsr.size)

    @property
    def hsr_median(self) -> float:
        """Return median HSR, or NaN if none are scored."""
        return float(np.median(self.hsr)) if self.hsr.size else float("nan")

    def retained_fraction(self, tau: float | None = None) -> float:
        """Return fraction of scored heads with HSR >= tau."""
        if not self.hsr.size:
            return float("nan")
        return float((self.hsr >= (self.tau if tau is None else tau)).mean())


def hsr_points(points: np.ndarray, transform=None
               ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Return paired visible heads, amodal exclusions, and source indices."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if transform is None:
        return pts, pts, 0, np.arange(len(pts))

    moved, kept = transform.apply(pts)
    indices = np.flatnonzero(kept)
    src = pts[kept]

    amodal = 0
    occluded = transform.occluded_mask
    if occluded is not None and not transform.drop_occluded:
        visible = ~np.asarray(occluded, dtype=bool)[kept]
        amodal = int((~visible).sum())
        moved, src = moved[visible], src[visible]
        indices = indices[visible]
    return moved, src, amodal, indices


def _luminance(img: np.ndarray) -> np.ndarray:
    """Return Rec.601 luma in the 0-255 range."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_RGB2GRAY)
    raise ValueError(f"expected (H, W) or (H, W, 3), got {arr.shape}")


def _sigmas(extent: float) -> tuple[float, float]:
    sigma_lo = max(HSR_SIGMA_LO_MIN_PX, HSR_SIGMA_LO_HEADS * extent)
    sigma_hi = max(HSR_SIGMA_HI_MIN_PX, HSR_SIGMA_HI_HEADS * extent)
    if sigma_hi <= sigma_lo:
        sigma_hi = sigma_lo * 2.0
    return sigma_lo, sigma_hi


def _bandpass(gray: np.ndarray, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_lo, borderType=cv2.BORDER_REFLECT)
    high = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_hi, borderType=cv2.BORDER_REFLECT)
    return low - high


def _noise_field(gray: np.ndarray) -> np.ndarray:
    """Return absolute Immerkaer filter response per pixel."""
    response = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=_NOISE_KERNEL,
                            borderType=cv2.BORDER_REFLECT)
    return np.abs(response)


def _scale_bins(extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return logarithmic head-extent bins and centers."""
    finite = extents[np.isfinite(extents) & (extents > 0)]
    if finite.size == 0:
        return np.zeros(len(extents), dtype=int), np.array([8.0])
    lo, hi = float(finite.min()), float(finite.max())
    if hi / max(lo, 1e-6) < 1.2:
        return np.zeros(len(extents), dtype=int), np.array([float(np.median(finite))])
    edges = np.geomspace(lo, hi, HSR_SCALE_BINS + 1)
    safe = np.where(np.isfinite(extents) & (extents > 0), extents, lo)
    index = np.clip(np.digitize(safe, edges[1:-1]), 0, HSR_SCALE_BINS - 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    return index, centres


def _tone_curve(src_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
    """Fit a monotone 256-entry clean-to-corrupted tone LUT."""
    lut = np.arange(256, dtype=np.float32)
    if src_vals.size < HSR_TONE_BINS * HSR_TONE_MIN_SAMPLES:
        return lut

    edges = np.linspace(0.0, 255.0, HSR_TONE_BINS + 1)
    which = np.clip(np.digitize(src_vals, edges[1:-1]), 0, HSR_TONE_BINS - 1)
    centres, mapped = [], []
    for b in range(HSR_TONE_BINS):
        sel = which == b
        if int(sel.sum()) < HSR_TONE_MIN_SAMPLES:
            continue
        centres.append(float(np.median(src_vals[sel])))
        mapped.append(float(np.median(dst_vals[sel])))
    if len(centres) < 3:
        return lut

    curve = np.maximum.accumulate(np.asarray(mapped, dtype=np.float64))
    knots = np.asarray(centres, dtype=np.float64)
    fitted = np.interp(lut, knots, curve)

    # Preserve contrast outside the fitted range.
    low, high = lut < knots[0], lut > knots[-1]
    fitted[low] = curve[0] + (lut[low] - knots[0])
    fitted[high] = curve[-1] + (lut[high] - knots[-1])
    return np.clip(fitted, 0.0, 255.0).astype(np.float32)


def head_structure_retention(
    clean: np.ndarray,
    corrupted: np.ndarray,
    points: np.ndarray,
    *,
    box_wh: np.ndarray | None = None,
    clean_points: np.ndarray | None = None,
    tau: float = RETENTION_TAU,
    match_photometry: bool = True,
) -> HSRResult:
    """Score head structure, excluding invalid or flat reference patches."""
    pts_dst = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts_src = (pts_dst if clean_points is None
               else np.asarray(clean_points, dtype=np.float64).reshape(-1, 2))
    if len(pts_src) != len(pts_dst):
        raise ValueError(
            f"clean_points has {len(pts_src)} rows but points has {len(pts_dst)}")

    empty = np.empty(0, dtype=np.float64)
    if len(pts_dst) == 0:
        return HSRResult(empty, 0, tau)

    gray_src = _luminance(clean)
    gray_dst = _luminance(corrupted)

    extents, _, _ = head_extent_px(pts_src, box_wh=box_wh)
    bin_index, bin_extents = _scale_bins(extents)

    boxes: list[tuple[int, tuple, tuple]] = []
    for i in range(len(pts_dst)):
        extent = float(bin_extents[int(bin_index[i])])
        half = int(round(np.clip(HSR_PATCH_HEADS * extent,
                                 HSR_MIN_PATCH_PX, HSR_MAX_PATCH_PX) / 2.0))
        src_box = _patch_bounds(pts_src[i], half, gray_src.shape)
        dst_box = _patch_bounds(pts_dst[i], half, gray_dst.shape)
        if src_box is None or dst_box is None:
            continue
        boxes.append((i, src_box, dst_box))

    if not boxes:
        return HSRResult(empty, len(pts_dst), tau)

    gray_ref = gray_src
    if match_photometry:
        src_pool, dst_pool = [], []
        for _, src_box, dst_box in boxes:
            a, c = _paired(gray_src, src_box, gray_dst, dst_box)
            if a is None:
                continue
            src_pool.append(a.ravel())
            dst_pool.append(c.ravel())
        if src_pool:
            lut = _tone_curve(np.concatenate(src_pool), np.concatenate(dst_pool))
            # Avoid quantization for near-identity tone curves.
            if np.abs(lut - np.arange(256, dtype=np.float32)).max() > 1.0:
                index = np.rint(np.clip(gray_src, 0.0, 255.0)).astype(np.uint8)
                gray_ref = np.clip(lut[index], 0.0, 255.0)

    noise_ref = _noise_field(gray_ref)
    noise_dst = _noise_field(gray_dst)
    noise_src = noise_ref if gray_ref is gray_src else _noise_field(gray_src)

    hsr_vals: list[float] = []

    for i, src_box, dst_box in boxes:
        b = int(bin_index[i])
        sigma_lo, sigma_hi = _sigmas(float(bin_extents[b]))
        gain = bandpass_noise_gain(sigma_lo, sigma_hi)

        # Pad by 3*sigma_hi for local filtering.
        pad = int(np.ceil(3.0 * sigma_hi))
        a, ok_a = _local_bandpass(gray_ref, src_box, pad, sigma_lo, sigma_hi)
        c, ok_c = _local_bandpass(gray_dst, dst_box, pad, sigma_lo, sigma_hi)
        if not (ok_a and ok_c):
            continue
        a, c = _paired_arrays(a, c)
        if a is None:
            continue

        sigma_ref = _sigma_in_band(noise_ref, src_box, gain)
        sigma_dst = max(sigma_ref, _sigma_in_band(noise_dst, dst_box, gain))

        d_ref = _signal_amplitude(a, sigma_ref) / sigma_ref
        d_dst = _signal_amplitude(c, sigma_dst) / sigma_dst
        ncc = _ncc(a, c)

        if d_ref >= HSR_MIN_REF_DETECTABILITY:
            ratio = float(np.clip(d_dst / d_ref, 0.0, 1.0))
        else:
            # Exclude flat heads; score lost matched structure as zero.
            if gray_ref is gray_src:
                src_patch, sigma_src = a, sigma_ref
            else:
                src_patch, _ = _local_bandpass(gray_src, src_box, pad,
                                               sigma_lo, sigma_hi)
                sigma_src = _sigma_in_band(noise_src, src_box, gain)
            if (_signal_amplitude(src_patch, sigma_src) / sigma_src
                    < HSR_MIN_REF_DETECTABILITY):
                continue
            ratio = 0.0

        hsr_vals.append(max(0.0, ncc) * ratio)

    return HSRResult(
        hsr=np.asarray(hsr_vals, dtype=np.float64),
        n_points=len(pts_dst),
        tau=tau,
    )


def _patch_bounds(point: np.ndarray, half: int, shape_hw: tuple[int, ...]):
    """Return in-frame patch bounds, or None."""
    height, width = shape_hw[0], shape_hw[1]
    cx, cy = int(round(float(point[0]))), int(round(float(point[1])))
    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    if (x1 - x0) < 3 or (y1 - y0) < 3:
        return None
    return x0, y0, x1, y1


def _crop(field: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return field[y0:y1, x0:x1]


def _paired(field_a, box_a, field_b, box_b):
    """Crop paired fields to a common shape."""
    return _paired_arrays(_crop(field_a, box_a), _crop(field_b, box_b))


def _paired_arrays(a, b):
    """Trim arrays to matching shapes, or return None."""
    if a is None or b is None:
        return None, None
    if a.shape != b.shape:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        if h < 3 or w < 3:
            return None, None
        a, b = a[:h, :w], b[:h, :w]
    return a, b


def _padded(field: np.ndarray, box: tuple[int, int, int, int], pad: int):
    """Return padded crop and inner bounds."""
    x0, y0, x1, y1 = box
    height, width = field.shape[0], field.shape[1]
    px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
    px1, py1 = min(width, x1 + pad), min(height, y1 + pad)
    return field[py0:py1, px0:px1], (x0 - px0, y0 - py0, x1 - px0, y1 - py0)


def _local_bandpass(gray: np.ndarray, box: tuple[int, int, int, int], pad: int,
                    sigma_lo: float, sigma_hi: float):
    """Filter a patch using its padded neighborhood."""
    region, inner = _padded(gray, box, pad)
    if region.shape[0] < 3 or region.shape[1] < 3:
        return None, False
    filtered = _bandpass(np.ascontiguousarray(region), sigma_lo, sigma_hi)
    x0, y0, x1, y1 = inner
    return filtered[y0:y1, x0:x1], True


def _sigma_in_band(noise: np.ndarray, box: tuple[int, int, int, int],
                   gain: float) -> float:
    """Estimate local in-band noise sigma with a floor."""
    return max(HSR_NOISE_FLOOR, _NOISE_SCALE * float(_crop(noise, box).mean())) * gain


def _signal_amplitude(bandpass_patch: np.ndarray, sigma_in_band: float) -> float:
    """Return noise-corrected band-pass RMS."""
    energy = float(bandpass_patch.std())
    return float(np.sqrt(max(0.0, energy * energy - sigma_in_band * sigma_in_band)))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Return zero-mean NCC, or 0 for constant patches."""
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    nx = float(np.sqrt((x * x).sum()))
    ny = float(np.sqrt((y * y).sum()))
    if nx < 1e-9 or ny < 1e-9:
        return 0.0
    return float((x * y).sum() / (nx * ny))
