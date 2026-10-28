"""Load crowd datasets into a common record format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

BUILTIN_DATASETS = ("sha", "shb", "jhu", "qnrf")

LAYOUTS: dict[str, tuple[str, str]] = {
    "sha": ("{split}_data/images", "{split}_data/ground-truth"),
    "shb": ("{split}_data/images", "{split}_data/ground-truth"),
    "jhu": ("{split}/images", "{split}/gt"),
    "qnrf": ("{Split}", "{Split}"),
}

TEST_SPLIT_SIZES = {"sha": 182, "shb": 316, "jhu": 1600, "qnrf": 334}


def source_dirs(dataset: str, root: str | Path, split: str) -> tuple[Path, Path]:
    """Resolve built-in image and annotation directories."""
    if dataset not in LAYOUTS:
        raise ValueError(
            f"{dataset!r} is not a built-in dataset ({', '.join(BUILTIN_DATASETS)}); "
            f"pass --images and --annotations for a custom one")
    fields = {"split": split, "Split": split.capitalize()}
    images, annotations = (Path(root) / t.format(**fields) for t in LAYOUTS[dataset])
    return images, annotations


def gt_file(dataset: str, gt_dir: Path, image_id: str) -> Path:
    """Return an image annotation path."""
    if dataset in ("sha", "shb"):
        return gt_dir / f"GT_{image_id}.mat"
    if dataset == "jhu":
        return gt_dir / f"{image_id}.txt"
    if dataset == "qnrf":
        return gt_dir / f"{image_id}_ann.mat"
    return gt_dir / f"{image_id}.json"


def image_size(img_path: str | Path) -> tuple[int, int]:
    """Read image dimensions from its header."""
    with Image.open(img_path) as im:
        w, h = im.size
    return int(h), int(w)


def _sha_points(gt_mat: Path) -> np.ndarray:
    m = sio.loadmat(str(gt_mat))
    info = m["image_info"]
    pts = info[0, 0][0, 0][0]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _load_shanghaitech(img_dir: Path, gt_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        gt_mat = gt_dir / f"GT_{stem}.mat"
        if not gt_mat.exists():
            raise FileNotFoundError(f"missing GT for {img_path}: {gt_mat}")
        pts = _sha_points(gt_mat)
        h, w = image_size(img_path)
        records.append({
            "img_path": str(img_path),
            "img_id": stem,
            "points": pts,
            "count": int(pts.shape[0]),
            "hw": (h, w),
            "meta": {},
        })
    return records


def _jhu_gt(gt_path: Path) -> dict[str, np.ndarray]:
    """Load JHU head coordinates and box sizes."""
    if gt_path.stat().st_size == 0:
        return {"points": np.empty((0, 2)), "box_wh": np.empty((0, 2))}
    arr = np.loadtxt(str(gt_path), ndmin=2)
    return {
        "points": arr[:, 0:2].astype(np.float64),
        "box_wh": arr[:, 2:4].astype(np.float64),
    }


def _load_jhu(img_dir: Path, gt_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        gt = _jhu_gt(gt_dir / f"{stem}.txt")
        h, w = image_size(img_path)
        records.append({
            "img_path": str(img_path),
            "img_id": stem,
            "points": gt["points"],
            "count": int(gt["points"].shape[0]),
            "hw": (h, w),
            "meta": {"box_wh": gt["box_wh"]},
        })
    return records


def _qnrf_points(ann_mat: Path) -> np.ndarray:
    """Load QNRF head coordinates from annPoints."""
    m = sio.loadmat(str(ann_mat))
    return np.asarray(m["annPoints"], dtype=np.float64).reshape(-1, 2)


def _load_qnrf(img_dir: Path, gt_dir: Path) -> list[dict[str, Any]]:
    if Path(img_dir) != Path(gt_dir):
        raise ValueError(
            f"QNRF stores images and annotations in one directory, but "
            f"images={img_dir} and annotations={gt_dir} differ.")
    records: list[dict[str, Any]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        ann = img_dir / f"{stem}_ann.mat"
        if not ann.exists():
            raise FileNotFoundError(f"missing QNRF annotation for {img_path}: {ann}")
        pts = _qnrf_points(ann)
        h, w = image_size(img_path)
        records.append({
            "img_path": str(img_path),
            "img_id": stem,
            "points": pts,
            "count": int(pts.shape[0]),
            "hw": (h, w),
            "meta": {},
        })
    return records


def _json_points(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("points", [])
    return np.asarray(payload, dtype=np.float64).reshape(-1, 2)


def _load_points_json(img_dir: Path, gt_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        ann = gt_dir / f"{stem}.json"
        if not ann.exists():
            raise FileNotFoundError(f"missing annotation for {img_path}: {ann}")
        pts = _json_points(ann)
        h, w = image_size(img_path)
        records.append({
            "img_path": str(img_path),
            "img_id": stem,
            "points": pts,
            "count": int(pts.shape[0]),
            "hw": (h, w),
            "meta": {},
        })
    return records


def load_split(dataset: str, img_dir: str | Path,
               gt_dir: str | Path) -> list[dict[str, Any]]:
    """Load annotated images using the dataset-specific parser."""
    img_dir, gt_dir = Path(img_dir), Path(gt_dir)
    if dataset in ("sha", "shb"):
        return _load_shanghaitech(img_dir, gt_dir)
    if dataset == "jhu":
        return _load_jhu(img_dir, gt_dir)
    if dataset == "qnrf":
        return _load_qnrf(img_dir, gt_dir)
    return _load_points_json(img_dir, gt_dir)
