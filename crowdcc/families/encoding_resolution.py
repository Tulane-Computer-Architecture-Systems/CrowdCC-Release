"""Compression, packet loss, resolution loss, and aliasing corruptions."""

from __future__ import annotations

import io

import av
import cv2
import numpy as np

from crowdcc.base import Context, GTTransform, LevelParams
from crowdcc.crowd_spacing import crowd_spacing_px
from crowdcc.image_ops import points_inside_mask
from crowdcc.registry import register

av.logging.set_level(av.logging.PANIC)

# spacing_resolution_loss
NATIVE_MEDIAN_SPACING_PX: dict[str, float] = {
    "sha": 17.23,
    "shb": 33.11,
    "jhu": 32.99,
    "qnrf": 34.78,
}

MIN_EFFECTIVE_SPACING_PX = 4.0

MIN_RESAMPLE_SCALE = 0.08

# h264_crf_compression, packet_loss_concealment
CHROMA_ALIGNMENT = 2

# packet_loss_concealment
H264_VCL_NAL_TYPES = (1, 5)

CONCEAL_DIFF_THRESHOLD = 6


def _pad_to_even(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad RGB frames to even dimensions; return the original shape."""
    height, width = rgb.shape[:2]
    pad_h = height % CHROMA_ALIGNMENT
    pad_w = width % CHROMA_ALIGNMENT
    if not (pad_h or pad_w):
        return rgb, (height, width)
    padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return padded, (height, width)


def _encode_video_frame(
    rgb: np.ndarray, codec: str, crf: int, slices: int
) -> tuple[bytes, tuple[int, int]]:
    """Encode a deterministic single-frame Annex-B H.264 stream."""
    padded, _ = _pad_to_even(rgb)
    options: dict[str, str] = {"crf": str(crf), "threads": "1"}
    if slices > 1:
        # Partition slices by macroblock count.
        mb_rows = (padded.shape[0] + 15) // 16
        mb_cols = (padded.shape[1] + 15) // 16
        per_slice = max(-(-(mb_rows * mb_cols) // slices), 1)
        options["x264-params"] = f"slice-max-mbs={per_slice}:sliced-threads=0"

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="h264")
    stream = container.add_stream(codec, rate=1)
    stream.width, stream.height = padded.shape[1], padded.shape[0]
    stream.pix_fmt = "yuv420p"
    stream.options = options
    frame = av.VideoFrame.from_ndarray(padded, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue(), padded.shape[:2]


def _decode_video_frame(
    stream_bytes: bytes, container_format: str, conceal: bool = False
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Decode a frame, optionally concealing missing H.264 slices."""
    options = {"err_detect": "ignore_err", "ec": "favor_inter"} if conceal else {}
    info: dict[str, object] = {"codec_name": None, "frames": 0, "packets": 0}
    try:
        container = av.open(io.BytesIO(stream_bytes), format=container_format,
                            options=options)
    except av.FFmpegError:
        return None, info
    frame_rgb: np.ndarray | None = None
    try:
        video = container.streams.video[0]
        video.codec_context.thread_type = "NONE"
        video.codec_context.thread_count = 1
        info["codec_name"] = video.codec_context.name
        for packet in container.demux(video):
            info["packets"] = int(info["packets"]) + 1
            for frame in packet.decode():
                info["frames"] = int(info["frames"]) + 1
                if frame_rgb is None:
                    frame_rgb = frame.to_ndarray(format="rgb24")
    except (av.FFmpegError, IndexError):
        pass
    finally:
        container.close()
    return frame_rgb, info


def _split_nal_units(stream_bytes: bytes) -> list[tuple[int, bytes]]:
    """Split an Annex-B stream into start-code-prefixed NAL units."""
    starts: list[tuple[int, int]] = []
    index, length = 0, len(stream_bytes)
    while index < length - 2:
        if stream_bytes[index] == 0 and stream_bytes[index + 1] == 0 \
                and stream_bytes[index + 2] == 1:
            if index > 0 and stream_bytes[index - 1] == 0:
                starts.append((index - 1, 4))
            else:
                starts.append((index, 3))
            index += 3
        else:
            index += 1
    units: list[tuple[int, bytes]] = []
    for position, (offset, code_length) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else length
        units.append((code_length, stream_bytes[offset:end]))
    return units


def _nal_unit_type(code_length: int, nal_bytes: bytes) -> int:
    """Read the H.264 NAL unit type."""
    return nal_bytes[code_length] & 0x1F


def _high_frequency_energy(rgb: np.ndarray) -> float:
    """Measure mean squared luma Laplacian energy."""
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.mean(cv2.Laplacian(luma, cv2.CV_32F, ksize=3) ** 2))


def _resample_round_trip(
    rgb: np.ndarray, scale: float, shrink: int, grow: int
) -> np.ndarray:
    """Downsample and restore the original image size."""
    height, width = rgb.shape[:2]
    small_w = max(2, int(round(width * scale)))
    small_h = max(2, int(round(height * scale)))
    small = cv2.resize(rgb, (small_w, small_h), interpolation=shrink)
    return cv2.resize(small, (width, height), interpolation=grow)


def _verify_spacing_resolution_loss(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Check output shape and report HF-energy ratio."""
    if src.shape != dst.shape:
        raise AssertionError(f"shape changed {src.shape} -> {dst.shape}; GT would be invalid")
    before = _high_frequency_energy(src)
    after = _high_frequency_energy(dst)
    return {"hf_energy_ratio": after / max(before, 1e-9)}


@register(
    name="spacing_resolution_loss",
    family="encoding_resolution",
    gt_impact="preserving",
    needs=["gt_points"],
    levels={
        "light": LevelParams(spacing_fraction=0.80),
        "moderate": LevelParams(spacing_fraction=0.70),
        "high": LevelParams(spacing_fraction=0.60),
    },
    verify=_verify_spacing_resolution_loss,
)
def spacing_resolution_loss(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply spacing-scaled area downsampling and linear upsampling."""
    native_median = NATIVE_MEDIAN_SPACING_PX.get(ctx.dataset)
    if native_median is None:
        native_median = ctx.meta.get("native_median_spacing_px")
        if native_median is None:
            raise ValueError(
                f"no native median crowd spacing for dataset {ctx.dataset!r}; "
                f"pass meta['native_median_spacing_px'] on the Context")
        native_median = float(native_median)
    measured = crowd_spacing_px(ctx.gt_points)
    spacing = native_median if not np.isfinite(measured) else measured

    fraction = p.spacing_fraction
    task_unit_target = fraction * min(spacing, native_median)
    target = max(task_unit_target, MIN_EFFECTIVE_SPACING_PX)
    scale = float(np.clip(target / spacing, MIN_RESAMPLE_SCALE, 1.0))

    out = _resample_round_trip(img, scale, cv2.INTER_AREA, cv2.INTER_LINEAR)
    return out, {
        "resample_scale": scale,
        "spacing_px_native": float(spacing),
        "spacing_px_effective": float(spacing * scale),
        "spacing_measured": float(np.isfinite(measured)),
        "spacing_target_capped": float(spacing < native_median),
        "spacing_fraction": float(fraction),
        "countability_floor_bound": float(task_unit_target < MIN_EFFECTIVE_SPACING_PX),
        "min_scale_bound": float(target / spacing < MIN_RESAMPLE_SCALE),
    }


def _blockiness(rgb: np.ndarray) -> float:
    """Measure the 8x8 grid-edge gradient ratio."""
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    columns = np.abs(np.diff(luma, axis=1))
    on_grid = columns[:, 7::8]
    off_grid = np.delete(columns, np.s_[7::8], axis=1)
    on_mean, off_mean = float(on_grid.mean()), float(off_grid.mean())
    if on_mean < 1e-6 and off_mean < 1e-6:
        return 1.0
    return on_mean / max(off_mean, 1e-9)


def _verify_h264_crf_compression(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Report HF-energy and blockiness without assertions; codec artifacts can raise HF energy."""
    return {
        "hf_energy_ratio": _high_frequency_energy(dst) / max(_high_frequency_energy(src), 1e-9),
        "blockiness_gain": _blockiness(dst) / max(_blockiness(src), 1e-9),
    }


@register(
    name="h264_crf_compression",
    family="encoding_resolution",
    gt_impact="preserving",
    needs=["ffmpeg"],
    levels={
        "light": LevelParams(codec="libx264", crf=30),
        "moderate": LevelParams(codec="libx264", crf=35),
        "high": LevelParams(codec="libx264", crf=40),
    },
    verify=_verify_h264_crf_compression,
)
def h264_crf_compression(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Round-trip a YUV420p frame through H.264 at the requested CRF."""
    height, width = img.shape[:2]
    stream_bytes, _ = _encode_video_frame(img, p.codec, p.crf, slices=1)
    decoded, info = _decode_video_frame(stream_bytes, "h264")
    if decoded is None:
        raise AssertionError(f"{p.codec} at crf={p.crf} produced no decodable frame")
    if info["codec_name"] != "h264":
        raise AssertionError(f"decoder reports codec {info['codec_name']!r}, not h264")

    # Check deterministic decoding.
    again, _ = _decode_video_frame(stream_bytes, "h264")
    if again is None or not np.array_equal(again, decoded):
        raise AssertionError("decoding the same stream twice gave different pixels")

    out = np.ascontiguousarray(decoded[:height, :width])
    return out, {
        "crf": float(p.crf),
        "bits_per_pixel": 8.0 * len(stream_bytes) / (height * width),
        "stream_bytes": float(len(stream_bytes)),
        "packets": float(info["packets"]),
        "frames": float(info["frames"]),
        "codec_is_h264": 1.0,
        "decode_reproducible": 1.0,
    }


def _uninitialised_buffer_fraction(rgb: np.ndarray) -> float:
    """Estimate green artifacts from unwritten YUV decoder buffers."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return float(((red < 20) & (green > 100) & (green < 170) & (blue < 20)).mean())


def _verify_packet_loss_concealment(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Check decoder artifacts and report slice-loss metrics."""
    row_error = np.abs(src.astype(np.int16) - dst.astype(np.int16)).mean(axis=(1, 2))
    concentration = float(row_error.max() / max(row_error.mean(), 1e-9))
    green_fraction = _uninitialised_buffer_fraction(dst)
    if green_fraction > 0.01:
        raise AssertionError(f"{green_fraction:.1%} of pixels left unconcealed")

    # Re-encode: dropped slice indices are not retained.
    stream_bytes, _ = _encode_video_frame(src, "libx264", p.crf, slices=p.slices)
    units = _split_nal_units(stream_bytes)
    n_slices = sum(1 for code_len, nal in units
                   if _nal_unit_type(code_len, nal) in H264_VCL_NAL_TYPES)
    requested = float(p.loss_rate)
    n_drop = int(np.clip(round(requested * n_slices), 1, max(n_slices - 1, 1)))

    intact, _ = _decode_video_frame(b"".join(nal for _, nal in units), "h264",
                                    conceal=True)
    concealed_fraction = float("nan")
    if intact is not None:
        intact = np.ascontiguousarray(intact[:dst.shape[0], :dst.shape[1]])
        concealed_fraction = float(
            (np.abs(dst.astype(np.int16) - intact.astype(np.int16)).max(axis=2)
             > CONCEAL_DIFF_THRESHOLD).mean())

    return {"row_error_concentration": concentration,
            "rows_damaged_fraction": float((row_error > 1.0).mean()),
            "uninitialised_green_fraction": green_fraction,
            "slices_encoded": float(n_slices),
            "slices_dropped": float(n_drop),
            "loss_rate_requested": requested,
            "loss_rate_achieved": n_drop / max(n_slices, 1),
            "concealed_area_fraction": concealed_fraction}


@register(
    name="packet_loss_concealment",
    family="encoding_resolution",
    gt_impact="altering",
    needs=["ffmpeg", "gt_points"],
    levels={
        "light": LevelParams(slices=32, crf=28, max_retries=6, loss_rate=2 / 32),
        "moderate": LevelParams(slices=32, crf=28, max_retries=6, loss_rate=4 / 32),
        "high": LevelParams(slices=32, crf=28, max_retries=6, loss_rate=8 / 32),
    },
    verify=_verify_packet_loss_concealment,
)
def packet_loss_concealment(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Drop H.264 slices and exclude concealed heads from GT."""
    height, width = img.shape[:2]
    stream_bytes, _ = _encode_video_frame(img, "libx264", p.crf, slices=p.slices)
    units = _split_nal_units(stream_bytes)
    vcl_indices = [i for i, (code_len, nal) in enumerate(units)
                   if _nal_unit_type(code_len, nal) in H264_VCL_NAL_TYPES]
    if len(vcl_indices) < 2:
        raise AssertionError(
            f"{len(vcl_indices)} VCL slice(s) encoded; at least 2 required"
        )
    loss_rate = float(p.loss_rate)
    n_drop = int(np.clip(round(loss_rate * len(vcl_indices)), 1, len(vcl_indices) - 1))

    # Isolate concealment error from codec distortion.
    intact, _ = _decode_video_frame(b"".join(nal for _, nal in units), "h264",
                                    conceal=True)
    if intact is None:
        raise AssertionError("loss-free reference stream did not decode")
    intact = np.ascontiguousarray(intact[:height, :width])

    for retry in range(int(p.max_retries) + 1):
        dropped = set(ctx.rng_image.choice(vcl_indices, size=n_drop, replace=False).tolist())
        damaged = b"".join(nal for i, (_, nal) in enumerate(units) if i not in dropped)
        decoded, info = _decode_video_frame(damaged, "h264", conceal=True)
        if decoded is not None:
            out = np.ascontiguousarray(decoded[:height, :width])
            concealed = (np.abs(out.astype(np.int16) - intact.astype(np.int16)).max(axis=2)
                         > CONCEAL_DIFF_THRESHOLD)
            return out, GTTransform(
                kind="occluded_mask", output_hw=(height, width),
                occluded_mask=points_inside_mask(ctx.gt_points, concealed, 0.5),
                drop_occluded=True,
            )
    raise AssertionError(
        f"no frame decoded after {int(p.max_retries) + 1} attempts "
        f"({n_drop}/{len(vcl_indices)} slices dropped)"
    )


def _bayer_mosaic_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Apply RGGB sampling and bilinear demosaicing."""
    height, width = rgb.shape[:2]
    mosaic = np.zeros((height, width), dtype=np.uint8)
    mosaic[0::2, 0::2] = rgb[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = rgb[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = rgb[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2RGB)


def _verify_moire_aliasing(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Report HF-energy excess relative to anti-aliased resampling."""
    if src.shape != dst.shape:
        raise AssertionError(f"shape changed {src.shape} -> {dst.shape}")
    control = _resample_round_trip(src, p.sample_factor, cv2.INTER_AREA, cv2.INTER_LINEAR)
    aliased = _high_frequency_energy(dst)
    anti_aliased = _high_frequency_energy(control)
    excess = aliased / max(anti_aliased, 1e-9)
    return {"aliasing_excess": excess,
            "hf_energy_ratio": aliased / max(_high_frequency_energy(src), 1e-9)}


@register(
    name="moire_aliasing",
    family="encoding_resolution",
    gt_impact="preserving",
    needs=[],
    levels={
        "light": LevelParams(sample_factor=0.70, cfa=True),
        "moderate": LevelParams(sample_factor=0.60, cfa=True),
        "high": LevelParams(sample_factor=0.50, cfa=True),
    },
    verify=_verify_moire_aliasing,
)
def moire_aliasing(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply optional Bayer sampling with nearest-down/cubic-up resizing."""
    sampled = _bayer_mosaic_roundtrip(img) if p.cfa else img
    out = _resample_round_trip(sampled, p.sample_factor, cv2.INTER_NEAREST, cv2.INTER_CUBIC)
    return out, {
        "sample_factor": float(p.sample_factor),
        "cfa": float(bool(p.cfa)),
    }
