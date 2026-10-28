"""Build the CrowdCC corruption benchmark from a crowd-counting dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from crowdcc import __version__, datasets, registry
from crowdcc.base import LEVELS
from crowdcc.depth import depth_dir
from crowdcc.generate import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_WORKERS,
    JOURNAL_NAME,
    DatasetBlock,
    canonical_dataset,
    density_spanning_sample,
    estimate_disk,
    execute,
    load_images,
    plan_jobs,
    print_cell_table,
    summarize,
    write_manifest,
)
from crowdcc.seeds import DEFAULT_MASTER_SEED


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True,
                   help="sha | shb | jhu | qnrf, or a name of your choice for a "
                        "custom dataset (then pass --images and --annotations)")
    p.add_argument("--root", type=Path, default=None,
                   help="root of a built-in dataset (e.g. .../ShanghaiTech/part_A, "
                        ".../jhu_crowd_v2.0, .../UCF-QNRF_ECCV18)")
    p.add_argument("--split", default="test",
                   help="dataset split (default: test, the benchmark split)")
    p.add_argument("--images", type=Path, default=None,
                   help="image directory (custom datasets, or to override --root)")
    p.add_argument("--annotations", type=Path, default=None,
                   help="annotation directory (custom datasets, or to override --root)")


def resolve_source(args) -> tuple[str, Path, Path]:
    """Resolve dataset and source directories."""
    dataset = canonical_dataset(args.dataset)
    if dataset in datasets.BUILTIN_DATASETS and args.root is not None:
        img_dir, gt_dir = datasets.source_dirs(dataset, args.root, args.split)
    elif args.images is None or args.annotations is None:
        need = ("--root (or --images and --annotations)"
                if dataset in datasets.BUILTIN_DATASETS
                else "--images and --annotations")
        raise SystemExit(f"{dataset}: pass {need}")
    else:
        img_dir, gt_dir = args.images, args.annotations
    return dataset, Path(args.images or img_dir), Path(args.annotations or gt_dir)


def resolve_corruptions(selector: str, exclude: str) -> list[str]:
    """Resolve corruption selections and exclusions."""
    try:
        chosen = {spec.name for spec in registry.resolve(selector)}
    except KeyError as exc:
        raise SystemExit(str(exc.args[0]))
    dropped = {t.strip() for t in exclude.split(",") if t.strip()}
    unknown = dropped - set(registry.names())
    if unknown:
        raise SystemExit(f"--exclude: unknown corruptions {sorted(unknown)}")
    return sorted(chosen - dropped)


def cmd_generate(args) -> int:
    registry.load_families()
    dataset, img_dir, gt_dir = resolve_source(args)
    names = resolve_corruptions(args.corruption, args.exclude)
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    unknown = sorted(set(levels) - set(LEVELS))
    if unknown:
        raise SystemExit(f"unknown levels {unknown}; allowed {list(LEVELS)}")

    out = Path(args.out)
    depth_root = Path(args.depth_root or out)
    dir_depth = depth_dir(depth_root, dataset, args.split)
    needs_depth = [n for n in names if "depth" in registry.get(n).needs]
    if needs_depth and not args.dry_run and not dir_depth.is_dir():
        raise SystemExit(
            f"{', '.join(needs_depth)} need depth maps, and {dir_depth} does not "
            f"exist.\nRun `crowdcc depth` for this dataset first (see README), or "
            f"leave them out with --exclude {','.join(needs_depth)}.")

    print(f"crowdcc     : {__version__}")
    print(f"dataset     : {dataset} / {args.split}")
    print(f"images      : {img_dir}")
    print(f"annotations : {gt_dir}")
    print(f"output      : {out}")
    print(f"corruptions : {len(names)}   levels: {levels}")
    print(f"workers     : {args.workers}"
          + (f"   sample: {args.sample} images" if args.sample else ""))
    if args.seed != DEFAULT_MASTER_SEED:
        print(f"seed        : {args.seed}  (NOT the benchmark seed "
              f"{DEFAULT_MASTER_SEED}; this is a different realization)")
    print()

    image_set = load_images(dataset, args.split, need_points=True,
                            img_dir=img_dir, gt_dir=gt_dir)
    expected = (datasets.TEST_SPLIT_SIZES.get(dataset)
                if args.split == "test" else None)
    if expected is not None and expected != len(image_set.records):
        print(f"  WARNING: the {dataset} test split should have {expected} images, "
              f"found {len(image_set.records)}. A different copy of the dataset "
              f"will not reproduce the published benchmark.", file=sys.stderr)

    started = time.time()
    report: dict[str, Any] = {
        "crowdcc": __version__, "dataset": dataset, "split": args.split,
        "corruptions": names, "levels": levels, "out": str(out),
        "mode": "sample" if args.sample else "full", "cells": [],
    }
    totals = {"done": 0, "skipped": 0, "failed": 0, "bytes": 0}
    failures: list[dict[str, Any]] = []

    for name in names:
        spec = registry.get(name)
        save_path = out / dataset / args.split / name
        block = DatasetBlock(
            dataset=dataset, split=args.split, image_path=img_dir, gt_path=gt_dir,
            save_path=save_path, gt_out_path=save_path / "gt",
            depth_path=dir_depth if "depth" in spec.needs else None,
            levels={lv: spec.params_for(lv) for lv in spec.level_names()},
        )
        if args.sample:
            records, sample_info = density_spanning_sample(image_set, name, args.sample_seed,
                                                n=args.sample)
        else:
            records, sample_info = list(image_set.records), None

        if args.estimate_only:
            estimate = estimate_disk(spec, image_set, levels, args.seed, block)
            report.setdefault("estimates", []).append(estimate)
            print(f"  [{name}] {estimate['n_images']} images -> "
                  f"{estimate['total_projected_gb']:.2f} GB over {len(levels)} levels")
            continue

        jobs = plan_jobs(spec, image_set, records, levels, block, args.seed,
                         args.overwrite, verify_on_skip=bool(args.sample))
        if args.dry_run:
            print(f"  [{name}] {len(jobs)} cells ({len(records)} images x "
                  f"{len(levels)} levels) -> {block.save_path}")
            continue

        results = execute(jobs, args.workers, f"{name}/{dataset}",
                          journal=block.save_path / JOURNAL_NAME)
        cell = summarize(spec, image_set, results, sample_info)
        report["cells"].append(cell)
        failures.extend(cell["failures"])
        for key in ("done", "skipped", "failed"):
            totals[key] += cell["counts"][key]
        totals["bytes"] += cell["bytes"]

        write_manifest(
            block.save_path, spec, image_set, levels,
            [r for r in results if r["status"] != "failed"],
            args.seed,
            params={lv: block.levels[lv].as_dict() for lv in levels},
            depth_dir=block.depth_path,
            extra={"gt_out_path": "gt",
                   **({"sample": sample_info} if sample_info else {})},
        )
        if args.verbose:
            print_cell_table(cell)

    elapsed = time.time() - started
    report["wall_clock_s"] = round(elapsed, 2)
    report["totals"] = totals

    # Verification failures retain image and GT; earlier failures do not.
    rejected = [f for f in failures if f.get("error_type") == "AssertionError"]
    errors = [f for f in failures if f.get("error_type") != "AssertionError"
              and f.get("image_id") != "corpus median"]
    ordering = [f for f in failures if f.get("image_id") == "corpus median"]
    report["rejected"], report["errors"], report["ordering"] = rejected, errors, ordering

    if not (args.dry_run or args.estimate_only):
        print("\n" + "=" * 78)
        print(f"done {totals['done']} | skipped {totals['skipped']} | "
              f"rejected {len(rejected)} | errors {len(errors)} | "
              f"{totals['bytes'] / 1e6:.1f} MB | {elapsed:.1f}s")
        kept = [f for f in rejected if f.get("outputs_written")]
        dropped = [f for f in rejected if not f.get("outputs_written")]
        if kept:
            print("\nflagged by their own mechanism check "
                  "(image + GT still written, listed in the report):")
            for key, n in _group(kept):
                print(f"  {n:5d}  {key}")
        if dropped:
            print("\nrejected before writing (no image or GT for these cells):")
            for key, n in _group(dropped):
                print(f"  {n:5d}  {key}")
        if ordering and not args.sample:
            print("\nnote: mean pixel change does not increase with severity on the "
                  "dataset median (expected for moire_aliasing; see docs/corruptions.md):")
            for failure in ordering:
                print(f"  {failure['corruption']}: {failure['error']}")
        if errors:
            print("\nERRORS (nothing was produced for these cells):")
            for key, n in _group(errors):
                print(f"  {n:5d}  {key}")
            for failure in errors[:5]:
                print(f"    {failure['corruption']}/{failure['level']}/"
                      f"{failure['image_id']}: {failure['error'][:150]}")
        print("=" * 78)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 1 if errors else 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _group(failures: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Group failures by corruption, level, and reason."""
    counts: dict[str, int] = {}
    for failure in failures:
        reason = re.split(r"[;,(]", failure["error"])[0].strip()
        reason = re.sub(r"\S*/\S+", "<path>", reason)
        reason = re.sub(r"\d+\.?\d*", "N", reason)
        if len(reason) > 88:
            reason = reason[:85] + "..."
        key = f"{failure['corruption']}/{failure['level']}: {reason}"
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def cmd_list(args) -> int:
    specs = list(registry.all_specs().values())
    if args.json:
        print(json.dumps([{
            "name": s.name, "family": s.family, "gt_impact": s.gt_impact,
            "needs": list(s.needs),
            "levels": {lv: s.params_for(lv).as_dict() for lv in s.level_names()},
        } for s in specs], indent=1))
        return 0
    print(f"{len(specs)} corruptions\n")
    print(f"  {'corruption':34s} {'family':20s} {'GT':10s} needs")
    for s in sorted(specs, key=lambda s: (s.family, s.name)):
        print(f"  {s.name:34s} {s.family:20s} {s.gt_impact:10s} "
              f"{','.join(s.needs) or '-'}")
    return 0


REFERENCE_FILE = Path(__file__).parent / "reference" / "csrnet.json"


def _packaged_reference(dataset: str, precision: str, report: dict) -> tuple[dict | None, str]:
    """Find a comparable packaged CSRNet reference."""
    payload = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    ref = payload["results"].get(dataset, {}).get(precision)
    if ref is None:
        return None, f"no packaged CSRNet reference for {dataset}/{precision}"
    if ref["n_images"] != report["n_images"]:
        return None, (f"the reference covers {ref['n_images']} images, this benchmark "
                      f"{report['n_images']}")
    if set(ref["per_corruption"]) != set(report["per_corruption"]):
        return None, "the reference covers a different set of corruptions"
    if any(set(values["levels"]) != set(LEVELS)
           for values in report["per_corruption"].values()):
        return None, "the reference requires all three severity levels for every corruption"
    expected_cells = {(name, level) for name in ref["per_corruption"] for level in LEVELS}
    actual_cells = {(cell["corruption"], cell["level"]) for cell in report["cells"]}
    if actual_cells != expected_cells or any(
        cell["n_images"] != ref["n_images"] for cell in report["cells"]
    ):
        return None, "the reference requires the full image count in every corruption/severity cell"
    return ref, f"CSRNet, {dataset}, {precision} (packaged)"


def cmd_evaluate(args) -> int:
    from crowdcc import evaluate as ev

    gt = ev.read_ground_truth(args.benchmark)
    predictions = ev.read_predictions(args.predictions)
    try:
        report = ev.score(predictions, gt)
    except ValueError as exc:
        raise SystemExit(str(exc))

    ref_label = "none"
    if args.reference and args.reference != "none":
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        ref_label = args.reference
    else:
        dataset = args.dataset or Path(args.benchmark).resolve().parent.name
        ref, ref_label = ((None, "disabled") if args.reference == "none"
                          else _packaged_reference(dataset, args.precision, report))
    if ref is not None:
        report["mce"] = ev.mce(report["per_corruption"], ref["per_corruption"])
    report["reference"] = ref_label

    families = defaultdict(list)
    for name, values in report["per_corruption"].items():
        family = registry.get(name).family if name in registry.names() else "other"
        families[family].append(values["mae_kept"])
    report["per_family"] = {f: sum(v) / len(v) for f, v in sorted(families.items())}

    print(f"images      : {report['n_images']}   corruptions: {report['n_corruptions']}")
    print(f"clean       : MAE {report['clean']['mae']:.2f}   RMSE {report['clean']['rmse']:.2f}")
    print(f"mcMAE       : {report['mcmae']:.2f}   (mcRMSE {report['mcrmse']:.2f})")
    print(f"RD          : {report['rd']:+.1f}%")
    print("mCE         : " + (f"{report['mce']:.3f}   vs {ref_label}" if report["mce"] is not None
                               else f"-   ({ref_label})"))
    if report["n_corruptions"] != 23:
        print(f"note        : {report['n_corruptions']} of the 23 corruptions were scored; "
              f"mcMAE/RD are not comparable to the paper's")
    print("\nper family (mean of E_c):")
    for family, value in report["per_family"].items():
        print(f"  {family:22s} {value:8.2f}")
    print("\nper corruption (E_c = mean over severities of MAE_kept), hardest first:")
    for name, values in sorted(report["per_corruption"].items(),
                               key=lambda kv: -kv[1]["mae_kept"]):
        print(f"  {name:34s} {values['mae_kept']:8.2f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    if args.cells:
        ev.write_cells_csv(args.cells, report)
        print(f"cells  -> {args.cells}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crowdcc",
        description="Build the CrowdCC corruption benchmark for crowd counting.",
    )
    parser.add_argument("--version", action="version", version=f"crowdcc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="corrupt a dataset split",
                       description=__doc__.split("\n\n")[0])
    add_dataset_args(g)
    g.add_argument("--out", type=Path, required=True, help="output root")
    g.add_argument("--corruption", default="all",
                   help="corruption name, family, 'all' (default), 'altering', "
                        "or a comma-separated list")
    g.add_argument("--exclude", default="", help="comma-separated corruptions to skip")
    g.add_argument("--levels", default=",".join(LEVELS),
                   help="comma-separated subset of light,moderate,high")
    g.add_argument("--depth-root", type=Path, default=None,
                   help="where `crowdcc depth` wrote its maps (default: --out)")
    g.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"worker processes (default {DEFAULT_WORKERS}); output is "
                        f"byte-identical at any value")
    g.add_argument("--sample", type=_positive_int, default=None, metavar="N",
                   help="only N images, drawn per corruption to span the density range")
    g.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    g.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED,
                   help=f"master seed (default {DEFAULT_MASTER_SEED} = the benchmark); "
                        f"change it only to draw an independent realization")
    g.add_argument("--overwrite", action="store_true",
                   help="regenerate cells even when a hash-matching output exists")
    g.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    g.add_argument("--estimate-only", action="store_true",
                   help="estimate disk use from ~20 images per level; write nothing")
    g.add_argument("--report", type=Path, default=None, help="write a JSON run report")
    g.add_argument("-v", "--verbose", action="store_true",
                   help="print a per-level verification table for every corruption")
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("depth", help="cache depth maps with Depth Anything 3 "
                                     "(needs the [depth] extra)")
    add_dataset_args(d)
    d.add_argument("--out", type=Path, required=True,
                   help="output root (same as `generate --out`)")
    d.add_argument("--checkpoint", default=None,
                   help="Depth Anything 3 checkpoint (default: the one the paper used)")
    d.add_argument("--device", default="cuda")
    d.add_argument("--limit", type=int, default=None, help="only the first N images")
    d.add_argument("--overwrite", action="store_true")
    d.add_argument("--validate-only", action="store_true",
                   help="skip inference; re-run the near/far validation checks")
    d.add_argument("--report", type=Path, default=None)
    d.set_defaults(func=_cmd_depth)

    e = sub.add_parser("evaluate", help="score predicted counts on a generated benchmark")
    e.add_argument("--benchmark", type=Path, required=True,
                   help="the <out>/<dataset>/<split> directory written by `crowdcc generate`")
    e.add_argument("--predictions", type=Path, required=True,
                   help="CSV with columns corruption,level,image_id,count (see README)")
    e.add_argument("--dataset", default=None,
                   help="dataset key for the packaged mCE reference (default: taken "
                        "from the benchmark path)")
    e.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                   help="which packaged CSRNet reference to use for mCE (default fp32)")
    e.add_argument("--reference", default=None,
                   help="a report JSON from `crowdcc evaluate --report` to use as the mCE "
                        "reference instead of the packaged CSRNet one, or 'none'")
    e.add_argument("--report", type=Path, default=None, help="write the full report as JSON")
    e.add_argument("--cells", type=Path, default=None,
                   help="write the per-(corruption, level) table as CSV")
    e.set_defaults(func=cmd_evaluate)

    ls = sub.add_parser("list", help="list the corruptions")
    ls.add_argument("--json", action="store_true",
                    help="include every severity parameter, as JSON")
    ls.set_defaults(func=cmd_list)
    return parser


def _cmd_depth(args) -> int:
    # Only this command requires torch.
    from crowdcc.depth_estimation import cmd_depth

    return cmd_depth(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
