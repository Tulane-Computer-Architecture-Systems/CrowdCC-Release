"""Generate corruption images, GT and manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from crowdcc import __version__, datasets, registry
from crowdcc.base import LEVELS, Context, CorruptionSpec, GTTransform, LevelParams
from crowdcc.depth import load_depth_dir
from crowdcc.image_ops import encode_png, read_image_rgb, to_uint8, write_png
from crowdcc.seeds import make_rngs, seeding_policy

MANIFEST_SCHEMA_VERSION = 1

SAMPLE_N_PER_DATASET = 10
SAMPLE_SPAN_PCTILES = (20.0, 80.0)
SAMPLE_MAX_REDRAWS = 50
DEFAULT_SAMPLE_SEED = 20260726

ESTIMATE_SAMPLE_SIZE = 20

DEFAULT_WORKERS = min(16, os.cpu_count() or 1)


@dataclass(frozen=True)
class ImageRecord:
    """Source image, ground truth, and metadata."""

    image_id: str
    src_path: Path
    gt_path: Path | None
    height: int
    width: int
    gt_count: int
    points: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageSet:
    """Labeled dataset split."""

    dataset: str
    split: str
    records: list[ImageRecord]


@dataclass(frozen=True)
class DatasetBlock:
    """Paths and severity settings for a corruption."""

    dataset: str
    split: str
    image_path: Path
    gt_path: Path
    save_path: Path
    gt_out_path: Path
    depth_path: Path | None
    levels: dict[str, LevelParams]


_DATASET_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def canonical_dataset(name: str) -> str:
    """Validate and normalize a dataset key."""
    key = name.strip()
    if key.lower() in datasets.BUILTIN_DATASETS:
        return key.lower()
    if not _DATASET_KEY.match(key):
        raise ValueError(
            f"dataset name {name!r} must be letters, digits, '_', '-' or '.'")
    return key


def load_images(dataset: str, split: str, need_points: bool,
                *, img_dir: Path, gt_dir: Path) -> ImageSet:
    """Load an image-id-sorted dataset split."""
    dataset = canonical_dataset(dataset)
    for label, directory in (("image_path", img_dir), ("gt_path", gt_dir)):
        if not Path(directory).is_dir():
            raise FileNotFoundError(
                f"{dataset}/{split}: {label} does not exist: {directory}")
    records = datasets.load_split(dataset, img_dir, gt_dir)
    if not records:
        raise FileNotFoundError(
            f"{dataset}/{split}: no images found under {img_dir}.")
    records.sort(key=lambda r: r["img_id"])
    # Built-ins use training spacing; custom datasets use split spacing.
    extra_meta: dict[str, Any] = {}
    if need_points and dataset not in datasets.BUILTIN_DATASETS:
        from crowdcc.crowd_spacing import dataset_median_spacing_px

        extra_meta["native_median_spacing_px"] = dataset_median_spacing_px(
            [np.asarray(r["points"], dtype=np.float64) for r in records])
    out = [
        ImageRecord(
            image_id=r["img_id"],
            src_path=Path(r["img_path"]),
            gt_path=datasets.gt_file(dataset, Path(gt_dir), r["img_id"]),
            height=int(r["hw"][0]),
            width=int(r["hw"][1]),
            gt_count=int(r["count"] or 0),
            points=(np.asarray(r["points"], dtype=np.float64) if need_points else None),
            meta={} if not need_points else {"box_wh": r["meta"].get("box_wh"), **extra_meta},
        )
        for r in records
    ]
    return ImageSet(dataset, split, out)


def density_spanning_sample(
    image_set: ImageSet, corruption: str, sample_seed: int,
    n: int = SAMPLE_N_PER_DATASET,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    """Sample images across the GT-count range."""
    records = image_set.records
    n = min(n, len(records))
    counts = np.array([r.gt_count for r in records], dtype=np.float64)
    lo, hi = np.percentile(counts, SAMPLE_SPAN_PCTILES)

    seed = int(
        hashlib.blake2b(
            f"sample|{sample_seed}|{image_set.dataset}|{image_set.split}|{corruption}".encode(),
            digest_size=8,
        ).digest().hex(),
        16,
    )
    rng = np.random.default_rng(seed)

    redraws = 0
    while True:
        idx = np.sort(rng.choice(len(records), size=n, replace=False))
        sampled = [records[i] for i in idx]
        sampled_counts = np.array([r.gt_count for r in sampled], dtype=np.float64)
        spans = sampled_counts.min() <= lo and sampled_counts.max() >= hi
        if spans or redraws >= SAMPLE_MAX_REDRAWS:
            break
        redraws += 1

    info = {
        "n": n,
        "sample_seed": sample_seed,
        "derived_seed": seed,
        "redraws": redraws,
        "span_ok": bool(spans),
        "gt_count_pctile_20_80": [float(lo), float(hi)],
        "sampled_gt_counts": [int(r.gt_count) for r in sampled],
        "image_ids": [r.image_id for r in sampled],
    }
    if not spans:
        info["warning"] = (
            f"draw does not span the {SAMPLE_SPAN_PCTILES} percentile range after "
            f"{redraws} redraws"
        )
    return sampled, info


@dataclass
class Job:
    """Picklable corruption-generation job."""

    dataset: str
    split: str
    corruption: str
    level: str
    image_id: str
    src_path: str
    gt_path: str | None
    out_dir: str
    master_seed: int
    overwrite: bool
    verify_on_skip: bool
    points: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Use block parameters, not registry defaults.
    params: LevelParams | None = None
    gt_out_dir: str | None = None
    depth_dir: str | None = None


def plan_jobs(
    spec: CorruptionSpec,
    image_set: ImageSet,
    records: Sequence[ImageRecord],
    levels: Sequence[str],
    block: "DatasetBlock",
    master_seed: int,
    overwrite: bool,
    verify_on_skip: bool,
) -> list[Job]:
    """Plan image-major jobs with block-level parameters."""
    missing = [lv for lv in levels if lv not in block.levels]
    if missing:
        raise KeyError(
            f"{spec.name}/{block.dataset}: levels {missing} were requested but are "
            f"not defined for it; defined: {sorted(block.levels)}")
    jobs: list[Job] = []
    for record in records:
        for level in levels:
            jobs.append(
                Job(
                    dataset=image_set.dataset,
                    split=image_set.split,
                    corruption=spec.name,
                    level=level,
                    image_id=record.image_id,
                    src_path=str(record.src_path),
                    gt_path=str(record.gt_path) if record.gt_path else None,
                    out_dir=str(block.save_path / level),
                    master_seed=master_seed,
                    overwrite=overwrite,
                    verify_on_skip=verify_on_skip,
                    points=record.points,
                    meta=record.meta,
                    params=block.levels[level],
                    gt_out_dir=str(block.gt_out_path / level),
                    depth_dir=str(block.depth_path) if block.depth_path else None,
                )
            )
    return jobs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RGB SSIM; return NaN for mismatched shapes."""
    from skimage.metrics import structural_similarity

    if a.shape != b.shape:
        return float("nan")
    if np.array_equal(a, b):
        return 1.0
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def countability_metrics(src_rgb: np.ndarray, dst_rgb: np.ndarray,
                         points: Any, box_wh: Any = None,
                         transform: Any = None) -> dict[str, Any]:
    """Report HSR metrics without asserting."""
    if points is None or len(points) < 4:
        return {}
    try:
        from crowdcc.countability import hsr_points, head_structure_retention

        moved, original, n_amodal, indices = hsr_points(points, transform)
        if len(moved) < 4:
            return {"hsr_median": float("nan"), "hsr_retained_fraction": float("nan"),
                    "hsr_n_amodal_excluded": float(n_amodal)}
        if box_wh is not None:
            box_wh = np.asarray(box_wh, dtype=np.float64).reshape(-1, 2)
            if len(box_wh) != len(points):
                raise ValueError("head boxes must match the original head points")
            box_wh = box_wh[indices]
        result = head_structure_retention(src_rgb, dst_rgb, moved,
                                          box_wh=box_wh, clean_points=original)
        return {
            "hsr_median": result.hsr_median,
            "hsr_retained_fraction": result.retained_fraction(),
            "hsr_n_scored": float(result.n_scored),
            "hsr_n_amodal_excluded": float(n_amodal),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).warning("HSR calculation failed: %s", error)
        return {"hsr_median": float("nan"), "hsr_retained_fraction": float("nan"),
                "hsr_error": error}


def verify_cell(
    spec: CorruptionSpec,
    src_rgb: np.ndarray,
    dst_rgb: np.ndarray,
    params: Any,
    ctx: Context,
    transform: GTTransform | None = None,
) -> dict[str, Any]:
    """Validate a corruption output and collect metrics."""
    if dst_rgb.dtype != np.uint8:
        raise AssertionError(f"output dtype is {dst_rgb.dtype}, expected uint8")
    if dst_rgb.ndim != 3 or dst_rgb.shape[2] != 3:
        raise AssertionError(f"output shape {dst_rgb.shape} is not (H, W, 3) RGB")
    if spec.gt_impact == "preserving" and dst_rgb.shape != src_rgb.shape:
        raise AssertionError(f"shape changed {src_rgb.shape} -> {dst_rgb.shape}")

    channel_std = dst_rgb.reshape(-1, 3).std(axis=0)
    if float(channel_std.max()) == 0.0:
        raise AssertionError("constant output image")

    delta = np.abs(src_rgb.astype(np.int16) - dst_rgb.astype(np.int16)) \
        if src_rgb.shape == dst_rgb.shape else None
    metrics: dict[str, Any] = {
        "mean_abs_delta": float(delta.mean()) if delta is not None else float("nan"),
        "max_abs_delta": float(delta.max()) if delta is not None else float("nan"),
        "ssim": _ssim(src_rgb, dst_rgb),
        "channel_std_min": float(channel_std.min()),
    }
    metrics.update(countability_metrics(
        src_rgb, dst_rgb, ctx.gt_points,
        box_wh=ctx.meta.get("box_wh"), transform=transform))

    if metrics["mean_abs_delta"] == 0.0:
        raise AssertionError(f"{spec.name}: output identical to input")

    if spec.verify is not None:
        metrics.update(spec.verify(src_rgb, dst_rgb, params, ctx))
    return metrics


def check_level_ordering(per_level_delta: dict[str, float]) -> tuple[bool, str]:
    """Check increasing pixel change across severity levels."""
    ordered = [lv for lv in LEVELS if lv in per_level_delta]
    if len(ordered) < 2:
        return True, "not checked (fewer than two levels generated)"
    values = [per_level_delta[lv] for lv in ordered]
    ok = all(b > a for a, b in zip(values, values[1:]))
    formatted = [f"{lv}={v:.4f}" for lv, v in zip(ordered, values)]
    return ok, ("increasing " + " < ".join(formatted) if ok
                else "NOT increasing: " + ", ".join(formatted))


def _prepare_output(raw: Any) -> tuple[np.ndarray, GTTransform | None, dict[str, Any]]:
    """Normalize the output image, GT transform, and metrics."""
    transform: GTTransform | None = None
    measurements: dict[str, Any] = {}
    if isinstance(raw, tuple):
        raw, extra = raw
        if isinstance(extra, GTTransform):
            transform = extra
        elif isinstance(extra, dict):
            measurements = extra
        elif extra is not None:
            raise AssertionError(
                f"unsupported second return element: {type(extra).__name__}"
            )
    array = np.asarray(raw)
    if array.dtype == np.uint8:
        return array, transform, measurements
    if not np.all(np.isfinite(array)):
        raise AssertionError("corruption produced NaN/Inf values")
    if array.min() < -1e-3 or array.max() > 1.0 + 1e-3:
        raise AssertionError(
            f"float output outside [0, 1]: [{array.min():.4f}, {array.max():.4f}]"
        )
    return to_uint8(array), transform, measurements


def _gt_payload(
    image_id: str,
    points: np.ndarray | None,
    transform: GTTransform | None,
    output_hw: tuple[int, int],
) -> dict[str, Any]:
    """Build and validate per-image GT metadata."""
    if points is None:
        raise AssertionError(f"{image_id}: no GT points loaded")
    if transform is None:
        return {
            "image_id": image_id,
            "kind": "identity",
            "output_hw": [int(output_hw[0]), int(output_hw[1])],
            "n_original": int(len(points)),
            "n_dropped": 0,
            "n_kept": int(len(points)),
            "kept_mask": [True] * len(points),
            "points_xy": np.asarray(points, dtype=np.float64).round(4).tolist(),
            "homography": None,
            "occluded_mask": None,
            "drop_occluded": False,
        }
    kept_points, kept_mask = transform.apply(points)
    n_dropped = int((~kept_mask).sum())
    if len(kept_points) != len(points) - n_dropped:
        raise AssertionError(
            f"count arithmetic violated: {len(kept_points)} kept != "
            f"{len(points)} original - {n_dropped} dropped"
        )
    height, width = transform.output_hw
    in_bounds = (
        (kept_points[:, 0] >= 0) & (kept_points[:, 0] < width)
        & (kept_points[:, 1] >= 0) & (kept_points[:, 1] < height)
    )
    if not bool(in_bounds.all()):
        raise AssertionError("transformed points fall outside the output image bounds")

    return {
        "image_id": image_id,
        "kind": transform.kind,
        "output_hw": [int(height), int(width)],
        "n_original": int(len(points)),
        "n_dropped": n_dropped,
        "n_kept": int(len(kept_points)),
        "kept_mask": kept_mask.astype(bool).tolist(),
        "points_xy": kept_points.round(4).tolist(),
        "homography": (transform.homography.tolist()
                       if transform.homography is not None else None),
        "occluded_mask": (np.asarray(transform.occluded_mask, bool).tolist()
                          if transform.occluded_mask is not None else None),
        "drop_occluded": bool(transform.drop_occluded),
    }


def _transform_from_gt(payload: dict[str, Any]) -> GTTransform | None:
    """Restore a GT transform from saved metadata."""
    if payload["kind"] == "identity":
        return None
    occluded = payload.get("occluded_mask")
    return GTTransform(
        kind=payload["kind"],
        output_hw=tuple(payload["output_hw"]),
        homography=(np.asarray(payload["homography"], np.float64)
                    if payload.get("homography") is not None else None),
        dropped_mask=~np.asarray(payload["kept_mask"], bool),
        occluded_mask=np.asarray(occluded, bool) if occluded is not None else None,
        drop_occluded=bool(payload.get("drop_occluded", False)),
    )


def _write_gt_payload(gt_out_dir: Path, image_id: str,
                      payload: dict[str, Any]) -> dict[str, Any]:
    """Write GT JSON and return manifest fields."""
    gt_out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = Path(gt_out_dir) / f"{image_id}.gt.json"
    gt_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return {k: payload[k] for k in ("kind", "n_original", "n_dropped", "n_kept")} | {
        "gt_file": gt_path.name
    }


def run_job(job: Job) -> dict[str, Any]:
    """Generate or reuse one cell and record the result."""
    import cv2

    cv2.setNumThreads(1)     # Avoid nested OpenCV threads.

    spec = registry.get(job.corruption)
    if job.params is None:
        raise AssertionError(f"{job.corruption}/{job.level}: job has no parameters")
    params = job.params
    src_path = Path(job.src_path)
    out_dir = Path(job.out_dir)
    dst_name = f"{job.image_id}.png"
    dst_path = out_dir / dst_name
    gt_file = Path(job.gt_out_dir) / f"{job.image_id}.gt.json" if job.gt_out_dir else None

    record: dict[str, Any] = {
        "image_id": job.image_id,
        "level": job.level,
        "dst_name": dst_name,
        "params_used": params.as_dict(),
    }
    try:
        src_sha = _sha256_file(src_path)
        record["src_sha256"] = src_sha
        record["gt_src_sha256"] = (
            _sha256_file(Path(job.gt_path)) if job.gt_path and Path(job.gt_path).exists()
            else None
        )
        record["master_seed"] = job.master_seed
        if "depth" in spec.needs and job.depth_dir:
            depth_file = Path(job.depth_dir) / f"{job.image_id}.npy"
            record["depth_sha256"] = _sha256_file(depth_file) if depth_file.exists() else None

        # Resume only with matching inputs and outputs.
        prior = _load_prior_entry(out_dir.parent, job.image_id, job.level)
        up_to_date = (
            not job.overwrite
            and prior is not None
            and all(prior.get(k) == record.get(k) for k in
                    ("src_sha256", "gt_src_sha256", "depth_sha256", "master_seed",
                     "params_used"))
            and dst_path.exists()
            and prior.get("dst_sha256") == _sha256_file(dst_path)
            and (gt_file is None or (gt_file.exists()
                                     and prior.get("gt_out_sha256") == _sha256_file(gt_file)))
        )
        if up_to_date:
            record.update(prior)
            record["status"] = "skipped"
            record["bytes"] = dst_path.stat().st_size
            if job.verify_on_skip:
                src_rgb = read_image_rgb(src_path)
                stored = read_image_rgb(dst_path)
                transform = (_transform_from_gt(json.loads(gt_file.read_text("utf-8")))
                             if gt_file is not None else None)
                prior_metrics = {k: v for k, v in prior.get("metrics", {}).items()
                                 if not k.startswith("hsr_")}
                record["metrics"] = {**prior_metrics, **verify_cell(
                    spec, src_rgb, stored, params,
                    _make_context(job, spec, src_rgb.shape[:2]), transform=transform,
                )}
            return record

        src_rgb = read_image_rgb(src_path)
        ctx = _make_context(job, spec, src_rgb.shape[:2])
        dst_rgb, transform, measurements = _prepare_output(spec.fn(src_rgb, params, ctx))

        if transform is not None and spec.gt_impact != "altering":
            raise AssertionError(
                f"{spec.name}: declared GT-preserving but returned a GTTransform"
            )
        if transform is None and spec.gt_impact == "altering":
            raise AssertionError(
                f"{spec.name}: declared GT-altering but returned no GTTransform"
            )
        # Validate GT before writing.
        gt_payload = (_gt_payload(job.image_id, job.points, transform, dst_rgb.shape[:2])
                      if gt_file is not None else None)

        out_dir.mkdir(parents=True, exist_ok=True)
        write_png(dst_path, dst_rgb)
        stored_rgb = read_image_rgb(dst_path)
        if not np.array_equal(stored_rgb, dst_rgb):
            raise AssertionError("cached image does not decode to the generated pixels")

        # Preserve GT for rejected outputs.
        record["gt"] = (_write_gt_payload(Path(job.gt_out_dir), job.image_id, gt_payload)
                        if gt_payload is not None else None)
        if gt_file is not None:
            record["gt_out_sha256"] = _sha256_file(gt_file)
        record["outputs_written"] = True

        record["metrics"] = verify_cell(spec, src_rgb, stored_rgb, params, ctx,
                                        transform=transform)
        if measurements:
            record["measurements"] = measurements
            record["metrics"].update(
                {k: float(v) for k, v in measurements.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )

        record["dst_sha256"] = _sha256_file(dst_path)
        record["bytes"] = dst_path.stat().st_size
        record["status"] = "done"
        return record

    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_type"] = type(exc).__name__
        record.setdefault("outputs_written", False)
        if not record["outputs_written"]:
            # Remove partial outputs on failure.
            for stale in (dst_path, gt_file):
                if stale is not None:
                    stale.unlink(missing_ok=True)
        return record


def _make_context(
    job: Job, spec: CorruptionSpec, shape_hw: tuple[int, int] | None = None
) -> Context:
    """Build a context with required per-job inputs."""
    rng_sensor, rng_image = make_rngs(
        job.dataset, job.image_id, job.corruption, job.level, job.master_seed
    )
    if "gt_points" in spec.needs and job.points is None:
        raise RuntimeError(f"{spec.name} needs gt_points but none were loaded")

    depth = None
    if "depth" in spec.needs:
        if job.depth_dir is None:
            raise RuntimeError(
                f"{spec.name} needs a depth map but the job carries no depth "
                f"directory; run `crowdcc depth` and pass --depth-root"
            )
        depth = load_depth_dir(
            job.depth_dir, job.image_id, expect_hw=shape_hw,
            what=f"{job.dataset}/{job.split}/{job.image_id}",
            rebuild_hint=f"--dataset {job.dataset} --split {job.split}",
        )

    # Pass only declared context inputs.
    return Context(
        image_id=job.image_id,
        dataset=job.dataset,
        level=job.level,
        rng_sensor=rng_sensor,
        rng_image=rng_image,
        gt_points=job.points if "gt_points" in spec.needs else None,
        depth=depth,
        meta=job.meta if "gt_points" in spec.needs else {},
    )


def manifest_path(corruption_dir: Path) -> Path:
    """Return the per-corruption manifest path."""
    return Path(corruption_dir) / "manifest.json"


JOURNAL_NAME = "manifest.journal.jsonl"
_PRIOR_CACHE: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}


def _load_prior_entry(
    corruption_dir: Path, image_id: str, level: str
) -> dict[str, Any] | None:
    """Load a prior cell entry from the manifest or journal."""
    path = manifest_path(corruption_dir)
    if path not in _PRIOR_CACHE:
        entries: dict[tuple[str, str], dict[str, Any]] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = {(e["image_id"], e["level"]): e for e in payload.get("images", [])}
        except (OSError, json.JSONDecodeError):
            pass
        entries.update(_read_journal(corruption_dir))
        _PRIOR_CACHE[path] = entries
    return _PRIOR_CACHE[path].get((image_id, level))


def _read_journal(corruption_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read valid entries from an interrupted run."""
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        lines = (Path(corruption_dir) / JOURNAL_NAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return entries
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries[(entry["image_id"], entry["level"])] = entry
    return entries


def _versions() -> dict[str, str]:
    import cv2
    import skimage

    return {
        "crowdcc": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit-image": skimage.__version__,
        "platform": platform.platform(),
    }


def _depth_provenance(depth_dir: Path | None) -> dict[str, Any]:
    """Read depth-model provenance or return an error."""
    from crowdcc.depth import load_depth_manifest_dir

    if depth_dir is None:
        return {"error": "no depth directory on the job"}
    try:
        manifest = load_depth_manifest_dir(depth_dir)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        key: manifest.get(key) for key in (
            "model", "checkpoint", "checkpoint_license", "convention",
            "normalization", "resize_interpolation", "dtype", "versions",
        )
    }


def write_manifest(
    corruption_dir: Path,
    spec: CorruptionSpec,
    image_set: ImageSet,
    levels: Sequence[str],
    records: Iterable[dict[str, Any]],
    master_seed: int,
    params: dict[str, Any] | None = None,
    depth_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Merge and write a sorted per-corruption manifest."""
    out_root = Path(corruption_dir)
    path = manifest_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            merged = {(e["image_id"], e["level"]): e for e in old.get("images", [])}
        except json.JSONDecodeError:
            merged = {}
    merged.update({k: {f: v for f, v in e.items() if f != "status"}
                   for k, e in _read_journal(out_root).items()})
    for entry in records:

        merged[(entry["image_id"], entry["level"])] = {
            k: v for k, v in entry.items() if k != "status"
        }

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": image_set.dataset,
        "split": image_set.split,
        "corruption": spec.name,
        "family": spec.family,
        "gt_impact": spec.gt_impact,
        "needs": list(spec.needs),
        "cache_format": "lossless PNG",
        "levels": (params if params is not None
                   else {lv: spec.params_for(lv).as_dict() for lv in levels}),
        "seeding": seeding_policy(master_seed),
        # Record depth-map provenance.
        **({"depth_provenance": _depth_provenance(depth_dir)}
           if "depth" in spec.needs else {}),
        "images_source": {
            "policy": "every annotated image in the split (see `sample` when sampled)",
            "n_images_available": len(image_set.records),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "images": [
            merged[k] for k in sorted(merged, key=lambda k: (k[0], LEVELS.index(k[1])))
        ],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8")
    (out_root / JOURNAL_NAME).unlink(missing_ok=True)
    return path


class Progress:
    """Track progress with tqdm or text output."""

    def __init__(self, total: int, label: str, use_tqdm: bool) -> None:
        self.total, self.label, self.done = total, label, 0
        self.bar = None
        self._next_report = 0.0
        if use_tqdm:
            from tqdm import tqdm

            self.bar = tqdm(total=total, desc=label, unit="img", leave=False)
        else:
            print(f"[{label}] {total} cells", flush=True)

    def update(self, n: int = 1) -> None:
        self.done += n
        if self.bar is not None:
            self.bar.update(n)
            return
        fraction = self.done / max(self.total, 1)
        if fraction >= self._next_report or self.done == self.total:
            print(f"[{self.label}] {self.done}/{self.total} ({fraction:6.1%})", flush=True)
            self._next_report = fraction + 0.10

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def estimate_disk(
    spec: CorruptionSpec, image_set: ImageSet, levels: Sequence[str], master_seed: int,
    block: "DatasetBlock",
) -> dict[str, Any]:
    """Estimate PNG cache size from sampled images."""
    n_total = len(image_set.records)
    step = max(1, n_total // ESTIMATE_SAMPLE_SIZE)
    sample = image_set.records[::step][:ESTIMATE_SAMPLE_SIZE]

    per_level: dict[str, dict[str, float]] = {}
    for level in levels:
        sizes: list[int] = []
        for record in sample:
            src_rgb = read_image_rgb(record.src_path)
            rng_sensor, rng_image = make_rngs(
                image_set.dataset, record.image_id, spec.name, level, master_seed
            )
            ctx = Context(
                image_id=record.image_id, dataset=image_set.dataset, level=level,
                rng_sensor=rng_sensor, rng_image=rng_image,
                gt_points=record.points,
                depth=(
                    load_depth_dir(block.depth_path, record.image_id,
                                   expect_hw=src_rgb.shape[:2])
                    if "depth" in spec.needs else None
                ),
                meta=record.meta,
            )
            dst_rgb, _, _ = _prepare_output(spec.fn(src_rgb, block.levels[level], ctx))
            sizes.append(len(encode_png(dst_rgb)))
        mean_bytes = float(np.mean(sizes))
        per_level[level] = {
            "sampled": len(sizes),
            "mean_bytes": mean_bytes,
            "projected_gb": mean_bytes * n_total / 1e9,
        }
    return {
        "dataset": image_set.dataset,
        "split": image_set.split,
        "corruption": spec.name,
        "n_images": n_total,
        "per_level": per_level,
        "total_projected_gb": sum(v["projected_gb"] for v in per_level.values()),
    }


def execute(jobs: list[Job], workers: int, label: str,
            journal: Path | None = None) -> list[dict[str, Any]]:
    """Run jobs with optional multiprocessing and journaling."""
    # Refresh the resume cache.
    _PRIOR_CACHE.clear()
    use_tqdm = sys.stderr.isatty()
    progress = Progress(len(jobs), label, use_tqdm)
    results: list[dict[str, Any]] = []
    log = None
    if journal is not None:
        journal.parent.mkdir(parents=True, exist_ok=True)
        log = journal.open("a", encoding="utf-8")

    def collect(result: dict[str, Any]) -> None:
        results.append(result)
        if log is not None and result["status"] != "failed":
            log.write(json.dumps(result) + "\n")
            log.flush()
        progress.update()

    try:
        if workers <= 1:
            for job in jobs:
                collect(run_job(job))
        else:
            # Spawn to avoid fork/OpenCV issues.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for result in pool.imap_unordered(run_job, jobs, chunksize=1):
                    collect(result)
    finally:
        progress.close()
        if log is not None:
            log.close()
    results.sort(key=lambda r: (r["image_id"], LEVELS.index(r["level"])))
    return results


def summarize(
    spec: CorruptionSpec,
    image_set: ImageSet,
    results: list[dict[str, Any]],
    sample_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize per-level metrics and failures."""
    counts = {"done": 0, "skipped": 0, "failed": 0}
    total_bytes = 0
    failures: list[dict[str, Any]] = []
    per_level: dict[str, list[dict[str, Any]]] = {lv: [] for lv in LEVELS}
    per_image_delta: dict[str, dict[str, float]] = {}

    for result in results:
        counts[result["status"]] += 1
        total_bytes += int(result.get("bytes", 0))
        if result["status"] == "failed":
            failures.append({
                "corruption": spec.name, "dataset": image_set.dataset,
                "level": result["level"], "image_id": result["image_id"],
                "error": result["error"],
                "error_type": result.get("error_type", "Exception"),
                "outputs_written": bool(result.get("outputs_written")),
            })
            continue
        metrics = result.get("metrics")
        if metrics:
            per_level[result["level"]].append(metrics)
            per_image_delta.setdefault(result["image_id"], {})[result["level"]] = (
                metrics["mean_abs_delta"]
            )

    # Enforce ordering on dataset medians only.
    corpus_delta = {
        level: float(np.median([m["mean_abs_delta"] for m in per_level[level]]))
        for level in LEVELS if per_level[level]
    }
    ordering_ok, ordering_detail = check_level_ordering(corpus_delta)
    per_image_violations = [
        image_id for image_id, deltas in per_image_delta.items()
        if not check_level_ordering(deltas)[0]
    ]
    if not ordering_ok:
        failures.append({
            "corruption": spec.name, "dataset": image_set.dataset, "level": "*",
            "image_id": "corpus median",
            "error": f"level ordering violated on the dataset median: {ordering_detail}",
        })

    rows = []
    for level in LEVELS:
        metrics_list = per_level[level]
        if not metrics_list:
            continue
        row = {
            "level": level,
            "n_images": len(metrics_list),
            "n_failed": sum(1 for f in failures if f["level"] == level),
            "mean_abs_delta": float(np.mean([m["mean_abs_delta"] for m in metrics_list])),
            "ssim": float(np.mean([m["ssim"] for m in metrics_list])),
        }
        # Include metrics missing from some images.
        row["n_hsr_errors"] = sum("hsr_error" in m for m in metrics_list)
        skip = {"mean_abs_delta", "max_abs_delta", "ssim", "channel_std_min", "hsr_error"}
        mechanism_keys = sorted({k for m in metrics_list for k in m} - skip)
        for key in mechanism_keys:
            values = [m[key] for m in metrics_list if key in m]
            if not values:
                continue
            row[key] = float(np.mean(values))
            if len(values) != len(metrics_list):
                row[f"{key}__n_images"] = len(values)
        rows.append(row)

    return {
        "corruption": spec.name,
        "dataset": image_set.dataset,
        "split": image_set.split,
        "counts": counts,
        "bytes": total_bytes,
        "rows": rows,
        "level_ordering_ok": ordering_ok,
        "level_ordering_detail": ordering_detail,
        "level_ordering_corpus_median": corpus_delta,
        "per_image_ordering_violations": len(per_image_violations),
        "per_image_ordering_violation_ids": sorted(per_image_violations)[:20],
        "n_images_checked": len(per_image_delta),
        "failures": failures,
        "sample": sample_info,
    }


def print_cell_table(cell: dict[str, Any]) -> None:
    """Print per-level verification metrics."""
    header = f"{cell['corruption']} / {cell['dataset']} / {cell['split']}"
    print(f"\n  {header}")
    print(f"  {'level':<9} {'n':>3} {'fail':>5} {'mean|dpix|':>11} {'SSIM':>7}"
          f"   mechanism")
    for row in cell["rows"]:
        mechanism = {k: round(v, 4) for k, v in row.items()
                     if k not in ("level", "n_images", "n_failed",
                                  "mean_abs_delta", "ssim")}
        print(f"  {row['level']:<9} {row['n_images']:>3} "
              f"{row['n_failed']:>5} {row['mean_abs_delta']:>11.4f} {row['ssim']:>7.4f}"
              f"   {mechanism or ''}")
    verdict = "PASS" if (cell["counts"]["failed"] == 0 and cell["level_ordering_ok"]) \
        else "FAIL"
    print(f"  level-ordering (dataset median): "
          f"{'ok' if cell['level_ordering_ok'] else 'VIOLATED'}: {cell['level_ordering_detail']}")
    if cell.get("n_images_checked"):
        print(f"  per-image ordering: {cell['n_images_checked'] - cell['per_image_ordering_violations']}"
              f"/{cell['n_images_checked']} images also monotone (informational)")
    print(f"  verdict: {verdict}")
    if cell["sample"]:
        sample = cell["sample"]
        print(f"  sample (seed {sample['sample_seed']}, redraws "
              f"{sample['redraws']}, span_ok {sample['span_ok']}): "
              f"{', '.join(sample['image_ids'])}")
