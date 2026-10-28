"""Score CrowdCC count predictions."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from crowdcc.base import LEVELS

CLEAN = "clean"
GT_DIR = "gt"
GT_SUFFIX = ".gt.json"


@dataclass
class GroundTruth:
    """Ground-truth counts for generated benchmark cells."""

    # Cell key maps to (kept, original) counts.
    cells: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    # Clean count by image ID.
    clean: dict[str, float] = field(default_factory=dict)

    def corruptions(self) -> list[str]:
        return sorted({c for c, _, _ in self.cells})

    def levels(self, corruption: str) -> list[str]:
        present = {lv for c, lv, _ in self.cells if c == corruption}
        return [lv for lv in LEVELS if lv in present]


def read_ground_truth(benchmark_dir: str | Path) -> GroundTruth:
    """Read benchmark ground truth; check original-count consistency."""
    root = Path(benchmark_dir)
    gt = GroundTruth()
    for corruption_dir in sorted(p for p in root.iterdir() if (p / GT_DIR).is_dir()):
        for level_dir in sorted((corruption_dir / GT_DIR).iterdir()):
            for path in sorted(level_dir.glob(f"*{GT_SUFFIX}")):
                record = json.loads(path.read_text(encoding="utf-8"))
                image_id = path.name[: -len(GT_SUFFIX)]
                n_kept, n_original = float(record["n_kept"]), float(record["n_original"])
                gt.cells[(corruption_dir.name, level_dir.name, image_id)] = (n_kept, n_original)
                known = gt.clean.setdefault(image_id, n_original)
                if known != n_original:
                    raise ValueError(
                        f"{path}: n_original={n_original} but another GT file of "
                        f"{image_id} says {known}; the benchmark directory mixes builds")
    if not gt.cells:
        raise FileNotFoundError(
            f"no {GT_SUFFIX} files under {root}; pass the <out>/<dataset>/<split> "
            f"directory written by `crowdcc generate`")
    return gt


def iter_images(benchmark_dir: str | Path) -> Iterator[tuple[str, str, str, Path]]:
    """Yield generated image paths and corruption identifiers."""
    root = Path(benchmark_dir)
    for corruption_dir in sorted(p for p in root.iterdir() if (p / GT_DIR).is_dir()):
        for level in LEVELS:
            level_dir = corruption_dir / level
            if level_dir.is_dir():
                for png in sorted(level_dir.glob("*.png")):
                    yield corruption_dir.name, level, png.stem, png


def read_predictions(path: str | Path) -> dict[tuple[str, str, str], float]:
    """Read count predictions from CSV."""
    out: dict[tuple[str, str, str], float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"corruption", "level", "image_id", "count"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            corruption = row["corruption"].strip()
            level = "" if corruption == CLEAN else row["level"].strip()
            key = (corruption, level, row["image_id"].strip())
            if key in out:
                raise ValueError(f"{path}:{line}: duplicate prediction for {key}")
            count = float(row["count"])
            if not math.isfinite(count):
                raise ValueError(f"{path}:{line}: count is not finite: {row['count']!r}")
            out[key] = count
    return out


def write_predictions(path: str | Path,
                      rows: Iterable[tuple[str, str, str, float]]) -> None:
    """Write count predictions to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["corruption", "level", "image_id", "count"])
        for corruption, level, image_id, count in rows:
            writer.writerow([corruption, "" if corruption == CLEAN else level,
                             image_id, f"{float(count):.6g}"])


def _mae_rmse(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    errors = [p - g for p, g in pairs]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return mae, rmse


def score(predictions: dict[tuple[str, str, str], float], gt: GroundTruth,
          reference: dict | None = None) -> dict:
    """Score predictions against exact benchmark coverage."""
    expected = set(gt.cells) | {(CLEAN, "", i) for i in gt.clean}
    missing = sorted(expected - set(predictions))
    extra = sorted(set(predictions) - expected)
    if missing or extra:
        raise ValueError(
            f"predictions do not match the benchmark: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")

    clean_mae, clean_rmse = _mae_rmse(
        [(predictions[(CLEAN, "", i)], n) for i, n in gt.clean.items()])

    by_cell: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for (corruption, level, image_id), (n_kept, n_original) in gt.cells.items():
        by_cell[(corruption, level)].append(
            (predictions[(corruption, level, image_id)], n_kept, n_original))

    cells = []
    for (corruption, level), rows in sorted(by_cell.items(),
                                            key=lambda kv: (kv[0][0], LEVELS.index(kv[0][1]))):
        mae, rmse = _mae_rmse([(p, k) for p, k, _ in rows])
        n_original = sum(o for _, _, o in rows)
        f_drop = 1.0 - sum(k for _, k, _ in rows) / n_original if n_original else 0.0
        keep = 1.0 - f_drop
        cells.append({
            "corruption": corruption, "level": level, "n_images": len(rows),
            "mae": mae, "rmse": rmse, "f_drop": f_drop,
            "mae_kept": mae / keep if keep > 0 else math.nan,
            "rmse_kept": rmse / keep if keep > 0 else math.nan,
        })

    per_corruption: dict[str, dict] = {}
    for corruption in gt.corruptions():
        mine = [c for c in cells if c["corruption"] == corruption]
        per_corruption[corruption] = {
            "levels": [c["level"] for c in mine],
            "mae_kept": sum(c["mae_kept"] for c in mine) / len(mine),
            "rmse_kept": sum(c["rmse_kept"] for c in mine) / len(mine),
        }

    mcmae = sum(v["mae_kept"] for v in per_corruption.values()) / len(per_corruption)
    mcrmse = sum(v["rmse_kept"] for v in per_corruption.values()) / len(per_corruption)
    report = {
        "n_images": len(gt.clean),
        "n_corruptions": len(per_corruption),
        "clean": {"mae": clean_mae, "rmse": clean_rmse},
        "mcmae": mcmae,
        "mcrmse": mcrmse,
        "rd": 100.0 * (mcmae - clean_mae) / clean_mae if clean_mae else math.nan,
        "rd_rmse": 100.0 * (mcrmse - clean_rmse) / clean_rmse if clean_rmse else math.nan,
        "mce": None,
        "per_corruption": per_corruption,
        "cells": cells,
    }
    if reference is not None:
        report["mce"] = mce(per_corruption, reference["per_corruption"])
    return report


def mce(per_corruption: dict[str, dict], reference: dict[str, dict]) -> float:
    """Compute mean corruption error relative to a reference."""
    if set(per_corruption) != set(reference):
        raise ValueError(
            "model and reference must cover the same corruptions: model-only "
            f"{sorted(set(per_corruption) - set(reference))}, reference-only "
            f"{sorted(set(reference) - set(per_corruption))}")
    ratios = [per_corruption[c]["mae_kept"] / reference[c]["mae_kept"] for c in per_corruption]
    return sum(ratios) / len(ratios)


def write_cells_csv(path: str | Path, report: dict) -> None:
    """Write per-corruption, per-level metrics to CSV."""
    columns = ["corruption", "level", "n_images", "mae", "rmse", "f_drop",
               "mae_kept", "rmse_kept"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for cell in report["cells"]:
            writer.writerow({k: cell[k] for k in columns})
