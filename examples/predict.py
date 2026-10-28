"""Save CrowdCC crowd-count predictions to CSV."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from crowdcc import datasets
from crowdcc.cli import add_dataset_args, resolve_source
from crowdcc.evaluate import CLEAN, iter_images, read_ground_truth, write_predictions
from crowdcc.image_ops import read_image_rgb


def load_callable(spec: str):
    """Load a function from a ``module:function`` reference."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--model must be module:function, got {spec!r}")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module_name), attr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="the <out>/<dataset>/<split> directory from `crowdcc generate`")
    add_dataset_args(parser)
    parser.add_argument("--model", required=True, help="module:function returning a count")
    parser.add_argument("--out", type=Path, required=True, help="predictions CSV to write")
    args = parser.parse_args(argv)

    count = load_callable(args.model)
    dataset, img_dir, gt_dir = resolve_source(args)
    clean_ids = set(read_ground_truth(args.benchmark).clean)

    rows = []
    records = [r for r in datasets.load_split(dataset, img_dir, gt_dir) if r["img_id"] in clean_ids]
    missing = clean_ids - {r["img_id"] for r in records}
    if missing:
        raise SystemExit(f"{len(missing)} benchmark images are not in {img_dir}, "
                         f"e.g. {sorted(missing)[:3]}")
    for record in records:
        rows.append((CLEAN, "", record["img_id"], count(read_image_rgb(record["img_path"]))))
    print(f"clean: {len(records)} images")

    n = 0
    for corruption, level, image_id, path in iter_images(args.benchmark):
        rows.append((corruption, level, image_id, count(read_image_rgb(path))))
        n += 1
        if n % 500 == 0:
            print(f"corrupted: {n} images", flush=True)
    print(f"corrupted: {n} images")

    write_predictions(args.out, rows)
    print(f"predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
