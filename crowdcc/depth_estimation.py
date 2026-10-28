"""Depth Anything 3 estimation and validation."""


from __future__ import annotations

import json
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Stub the unused DA3 exporter to avoid optional dependencies.
_EXPORT_STUB = types.ModuleType("depth_anything_3.utils.export")
_EXPORT_STUB.export = lambda *a, **k: (_ for _ in ()).throw(
    NotImplementedError("DA3 export is stubbed: only prediction.depth is used")
)
sys.modules.setdefault("depth_anything_3.utils.export", _EXPORT_STUB)

from crowdcc import depth as depth_io  # noqa: E402
from crowdcc.crowd_spacing import head_diameter_px, point_spacing_px  # noqa: E402
from crowdcc.generate import ImageRecord, load_images  # noqa: E402

DEFAULT_CHECKPOINT = "depth-anything/DA3-GIANT-1.1"
CHECKPOINT_LICENSES = {
    "depth-anything/DA3-GIANT-1.1": "CC BY-NC 4.0",
    "depth-anything/DA3-GIANT": "CC BY-NC 4.0",
    "depth-anything/DA3-LARGE-1.1": "Apache-2.0",
    "depth-anything/DA3-LARGE": "Apache-2.0",
    "depth-anything/DA3-BASE": "Apache-2.0",
    "depth-anything/DA3-SMALL": "Apache-2.0",
}

NORM_PERCENTILES = (1.0, 99.0)
RESIZE_INTERPOLATION = "INTER_LINEAR"
CONVENTION_MIN_FRACTION = 0.75
SANITY_MAX_RHO = -0.3
SANITY_MIN_POINTS = 20
SANITY_MIN_PASS_FRACTION = 0.5


def load_model(checkpoint: str, device: str):
    """Load a DA3 checkpoint and return model metadata."""
    import torch
    from depth_anything_3.api import DepthAnything3

    started = time.time()
    model = DepthAnything3.from_pretrained(checkpoint)
    model = model.to(device=torch.device(device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "model": "Depth-Anything-3",
        "checkpoint": checkpoint,
        "checkpoint_license": CHECKPOINT_LICENSES.get(checkpoint, "unknown"),
        "parameters": int(n_params),
        "device": device,
        "load_seconds": round(time.time() - started, 2),
    }
    return model, torch, info


def estimate_depth(model, image_path: Path, hw: tuple[int, int]) -> dict[str, Any]:
    """Estimate normalized depth and diagnostics for one image."""
    import cv2

    prediction = model.inference([str(image_path)])
    raw = np.asarray(prediction.depth, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[0]
    native = cv2.resize(raw, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    low, high = np.percentile(native, NORM_PERCENTILES)
    span = max(float(high - low), 1e-6)
    normalized = np.clip((native - low) / span, 0.0, 1.0)
    height = normalized.shape[0]
    top = float(normalized[: height // 3].mean())
    bottom = float(normalized[2 * height // 3:].mean())
    confidence = getattr(prediction, "conf", None)
    return {
        "depth": normalized,
        "inference_hw": [int(raw.shape[0]), int(raw.shape[1])],
        "raw_p1": float(low),
        "raw_p99": float(high),
        "raw_min": float(native.min()),
        "raw_max": float(native.max()),
        "top_third_mean": top,
        "bottom_third_mean": bottom,
        "bottom_is_nearer": bool(bottom < top),
        "conf_median": (float(np.median(np.asarray(confidence)))
                        if confidence is not None else None),
    }


def generate(
    model, records: Sequence[ImageRecord], target: Path,
    dataset: str, split: str, model_info: dict[str, Any], overwrite: bool,
) -> dict[str, Any]:
    """Cache depth maps and write the manifest."""
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / depth_io.DEPTH_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        prior_ckpt = json.loads(manifest_path.read_text(encoding="utf-8")).get("checkpoint")
        if prior_ckpt != model_info.get("checkpoint"):
            raise RuntimeError(
                f"{target} holds maps from checkpoint {prior_ckpt!r}; refusing to add maps "
                f"from {model_info.get('checkpoint')!r}. Use --overwrite or another --out.")
    entries: list[dict[str, Any]] = []
    done = skipped = 0
    total_bytes = 0
    started = time.time()
    for index, record in enumerate(records, 1):
        path = target / f"{record.image_id}.npy"
        if path.exists() and not overwrite:
            skipped += 1
            total_bytes += path.stat().st_size
            continue
        result = estimate_depth(model, record.src_path, (record.height, record.width))
        np.save(path, result.pop("depth").astype(depth_io.DEPTH_DTYPE))
        total_bytes += path.stat().st_size
        entries.append({"image_id": record.image_id,
                        "hw": [record.height, record.width], **result})
        done += 1
        if index % 50 == 0 or index == len(records):
            rate = index / max(time.time() - started, 1e-6)
            print(f"  [{dataset}/{split}] {index}/{len(records)}  {rate:.2f} img/s")
    manifest = {
        "dataset": dataset,
        "split": split,
        "n_images": len(records),
        "n_generated": done,
        "n_skipped": skipped,
        "bytes": total_bytes,
        "wall_clock_s": round(time.time() - started, 2),
        "convention": "0 = near, 1 = far",
        "normalization": (f"per image, clip((raw - p{NORM_PERCENTILES[0]:g}) / "
                          f"(p{NORM_PERCENTILES[1]:g} - p{NORM_PERCENTILES[0]:g}), 0, 1)"),
        "resize_interpolation": RESIZE_INTERPOLATION,
        "dtype": str(np.dtype(depth_io.DEPTH_DTYPE)),
        **model_info,
        "versions": _versions(),
        "images": entries,
    }
    if manifest_path.exists() and not overwrite:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        known = {e["image_id"] for e in entries}
        manifest["images"] = [e for e in prior.get("images", [])
                              if e["image_id"] not in known] + entries
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    import torch

    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__}
    try:
        versions["depth_anything_3"] = version("depth_anything_3")
    except PackageNotFoundError:
        versions["depth_anything_3"] = "unknown"
    return versions


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Return Spearman rho, or NaN for insufficient or constant data."""
    import warnings
    from scipy.stats import ConstantInputWarning, spearmanr

    if len(a) < SANITY_MIN_POINTS:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def validate(
    dataset: str, split: str, records: Sequence[ImageRecord], target: Path,
) -> dict[str, Any]:
    """Validate depth convention and head-scale correlations."""
    spacing_rhos: list[float] = []
    head_rhos: list[float] = []
    bottom_nearer = 0
    checked = 0
    for record in records:
        depth = depth_io.load_depth_dir(target, record.image_id,
                                        expect_hw=(record.height, record.width))
        height = depth.shape[0]
        bottom_nearer += int(depth[2 * height // 3:].mean() < depth[: height // 3].mean())
        checked += 1
        points = record.points
        if points is None or len(points) < SANITY_MIN_POINTS:
            continue
        cols = np.clip(np.round(points[:, 0]).astype(int), 0, depth.shape[1] - 1)
        rows = np.clip(np.round(points[:, 1]).astype(int), 0, depth.shape[0] - 1)
        depth_at_head = depth[rows, cols]
        spacing = point_spacing_px(points)
        if len(spacing) == len(depth_at_head):
            spacing_rhos.append(_spearman(spacing, depth_at_head))
        box_wh = record.meta.get("box_wh")
        if box_wh is not None and len(box_wh) == len(depth_at_head):
            diameters = head_diameter_px(np.asarray(box_wh))
            usable = np.isfinite(diameters) & (diameters > 0)
            if usable.sum() >= SANITY_MIN_POINTS:
                head_rhos.append(_spearman(diameters[usable], depth_at_head[usable]))

    def summarize(values: list[float]) -> dict[str, Any]:
        clean = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
        if clean.size == 0:
            return {"n": 0, "median_rho": None, "pass_fraction": None}
        return {"n": int(clean.size),
                "median_rho": float(np.median(clean)),
                "p90_rho": float(np.percentile(clean, 90)),
                "pass_fraction": float((clean < SANITY_MAX_RHO).mean())}

    report = {
        "dataset": dataset, "split": split, "n_images": checked,
        "bottom_third_nearer_fraction": bottom_nearer / max(checked, 1),
        "spacing_vs_depth": summarize(spacing_rhos),
        "head_size_vs_depth": summarize(head_rhos),
    }
    report["convention_ok"] = (
        report["bottom_third_nearer_fraction"] >= CONVENTION_MIN_FRACTION)
    spacing = report["spacing_vs_depth"]
    report["sanity_ok"] = bool(
        spacing["median_rho"] is not None
        and spacing["median_rho"] < SANITY_MAX_RHO
        and spacing["pass_fraction"] >= SANITY_MIN_PASS_FRACTION)
    return report


def print_validation(report: dict[str, Any]) -> None:
    spacing, heads = report["spacing_vs_depth"], report["head_size_vs_depth"]
    print(f"\n  {report['dataset']}/{report['split']}  ({report['n_images']} images)")
    print(f"    bottom third nearer than top : "
          f"{report['bottom_third_nearer_fraction']:.3f} of images  "
          f"{'ok' if report['convention_ok'] else 'CONVENTION SUSPECT'}")
    if spacing["median_rho"] is not None:
        print(f"    rho(crowd_spacing_px, depth) : median {spacing['median_rho']:+.3f}  "
              f"({spacing['pass_fraction']:.2f} of {spacing['n']} images < {SANITY_MAX_RHO})"
              f"  {'PASS' if report['sanity_ok'] else 'FAIL'}")
    if heads["median_rho"] is not None:
        print(f"    rho(head diameter, depth)    : median {heads['median_rho']:+.3f}  "
              f"({heads['pass_fraction']:.2f} of {heads['n']} images < {SANITY_MAX_RHO})"
              f"   [JHU head-box measurements]")


def cmd_depth(args) -> int:
    from crowdcc.cli import resolve_source

    dataset, img_dir, gt_dir = resolve_source(args)
    image_set = load_images(dataset, args.split, need_points=True,
                            img_dir=img_dir, gt_dir=gt_dir)
    records = list(image_set.records)[: args.limit] if args.limit else list(image_set.records)
    target = depth_io.depth_dir(args.out, dataset, args.split)
    print(f"dataset    : {dataset} / {args.split}  ({len(records)} images)")
    print(f"depth maps : {target}")
    if not args.validate_only:
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        model, _, model_info = load_model(checkpoint, args.device)
        print(f"model      : {model_info['model']} {model_info['checkpoint']} "
              f"({model_info['parameters'] / 1e9:.2f}B, licence "
              f"{model_info['checkpoint_license']})")
        if checkpoint != DEFAULT_CHECKPOINT:
            print(f"WARNING: the published benchmark used {DEFAULT_CHECKPOINT}")
        manifest = generate(model, records, target, dataset, args.split,
                            model_info, args.overwrite)
        print(f"  {manifest['n_generated']} generated, {manifest['n_skipped']} "
              f"skipped, {manifest['bytes'] / 1e9:.2f} GB, {manifest['wall_clock_s']}s")
    report = validate(dataset, args.split, records, target)
    print_validation(report)
    failed = not (report["sanity_ok"] and report["convention_ok"])
    if failed:
        print("\nDepth validation failed. Check the convention and correlation "
              "results above before generating depth-dependent corruptions.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 1 if failed else 0
